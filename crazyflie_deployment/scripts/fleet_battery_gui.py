#!/usr/bin/env python3
"""
Standalone PyQt6 battery monitor for a Crazyswarm2 Crazyflie fleet.

Usage:
    python3 scripts/fleet_battery_gui.py
    python3 scripts/fleet_battery_gui.py --config config/crazyflies.yaml --all
"""

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


DEFAULT_CONFIG = (
    "/home/xxinyue/swarm_life/crazyflie_deployment/config/crazyflies.yaml"
)
READ_TIMEOUT_S = 6.0
LOG_PERIOD_MS = 100
LOG_DURATION_S = 1.0


@dataclass(frozen=True)
class Drone:
    name: str
    uri: str
    enabled: bool
    drone_type: str
    voltage_warning: float
    voltage_critical: float


@dataclass(frozen=True)
class BatteryResult:
    name: str
    voltage: Optional[float]
    status: str
    detail: str = ""


def load_drones(config_path: str, include_all: bool) -> list[Drone]:
    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    robots = data.get("robots") or {}
    robot_types = data.get("robot_types") or {}
    drones: list[Drone] = []

    for name, robot in robots.items():
        enabled = bool(robot.get("enabled", False))
        if not include_all and not enabled:
            continue

        drone_type = str(robot.get("type", ""))
        type_data = robot_types.get(drone_type) or {}
        battery = type_data.get("battery") or {}
        warning = float(battery.get("voltage_warning", 3.8))
        critical = float(battery.get("voltage_critical", 3.7))

        uri = robot.get("uri")
        if not uri:
            continue

        drones.append(
            Drone(
                name=str(name),
                uri=str(uri),
                enabled=enabled,
                drone_type=drone_type,
                voltage_warning=warning,
                voltage_critical=critical,
            )
        )

    return drones


def read_voltage_once(uri: str) -> Optional[float]:
    samples: list[float] = []

    def cb(_timestamp, data, _logconf):
        value = data.get("pm.vbat")
        if value is not None:
            samples.append(float(value))

    with SyncCrazyflie(uri) as scf:
        log_config = LogConfig(name="bat", period_in_ms=LOG_PERIOD_MS)
        log_config.add_variable("pm.vbat", "float")
        scf.cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(cb)
        log_config.start()
        time.sleep(LOG_DURATION_S)
        log_config.stop()

    if not samples:
        return None
    return sum(samples) / len(samples)


def read_voltage_with_timeout(uri: str, timeout_s: float = READ_TIMEOUT_S) -> BatteryResult:
    result_queue: queue.Queue[tuple[str, Optional[float], str]] = queue.Queue(maxsize=1)

    def target():
        try:
            voltage = read_voltage_once(uri)
            if voltage is None:
                result_queue.put(("OFFLINE", None, "no voltage samples"))
            else:
                result_queue.put(("OK", voltage, ""))
        except BaseException as exc:  # cflib/libusb failures should not kill the sweep.
            result_queue.put(("ERROR", None, f"{type(exc).__name__}: {exc}"))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    try:
        status, voltage, detail = result_queue.get(timeout=timeout_s)
        return BatteryResult(name="", voltage=voltage, status=status, detail=detail)
    except queue.Empty:
        return BatteryResult(name="", voltage=None, status="TIMEOUT", detail="read timed out")


class PollWorker(QThread):
    row_started = pyqtSignal(str)
    row_result = pyqtSignal(object)
    sweep_done = pyqtSignal(float)

    def __init__(self, drones: list[Drone]):
        super().__init__()
        self._drones = drones

    def run(self):
        started = time.monotonic()
        for drone in self._drones:
            self.row_started.emit(drone.name)
            result = read_voltage_with_timeout(drone.uri)
            self.row_result.emit(
                BatteryResult(
                    name=drone.name,
                    voltage=result.voltage,
                    status=result.status,
                    detail=result.detail,
                )
            )
        self.sweep_done.emit(time.monotonic() - started)


class FleetBatteryWindow(QMainWindow):
    def __init__(self, config_path: str, include_all: bool):
        super().__init__()
        self.config_path = config_path
        self.include_all = include_all
        self.drones: list[Drone] = []
        self.worker: Optional[PollWorker] = None

        self.setWindowTitle("Crazyflie Fleet Battery")
        self.resize(1050, 520)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.start_sweep)

        self.auto_refresh = QCheckBox("Auto-refresh")
        self.auto_refresh.stateChanged.connect(self.on_auto_refresh_changed)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(10, 3600)
        self.interval_spin.setValue(30)
        self.interval_spin.setSuffix(" s")
        self.interval_spin.valueChanged.connect(self.on_interval_changed)

        self.status_label = QLabel("Last sweep: never")

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.auto_refresh)
        toolbar.addWidget(self.interval_spin)
        toolbar.addStretch(1)
        toolbar.addWidget(self.status_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "URI", "Enabled", "Voltage", "Status"])
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout()
        layout.addLayout(toolbar)
        layout.addWidget(self.table)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_sweep)
        self.reload_table()

    def reload_table(self):
        try:
            self.drones = load_drones(self.config_path, self.include_all)
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            self.drones = []

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.drones))
        for row, drone in enumerate(self.drones):
            self.table.setItem(row, 0, QTableWidgetItem(drone.name))
            self.table.setItem(row, 1, QTableWidgetItem(drone.uri))
            self.table.setItem(row, 2, QTableWidgetItem("yes" if drone.enabled else "no"))
            self.table.setItem(row, 3, QTableWidgetItem(""))
            self.table.setItem(row, 4, QTableWidgetItem("PENDING"))
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def find_row(self, name: str) -> int:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == name:
                return row
        return -1

    def drone_by_name(self, name: str) -> Optional[Drone]:
        for drone in self.drones:
            if drone.name == name:
                return drone
        return None

    def start_sweep(self):
        if self.worker and self.worker.isRunning():
            return

        self.reload_table()
        if not self.drones:
            self.status_label.setText("Last sweep: no drones")
            return

        self.refresh_button.setEnabled(False)
        self.status_label.setText("Last sweep: running...")
        self.worker = PollWorker(self.drones)
        self.worker.row_started.connect(self.on_row_started)
        self.worker.row_result.connect(self.on_row_result)
        self.worker.sweep_done.connect(self.on_sweep_done)
        self.worker.finished.connect(lambda: self.refresh_button.setEnabled(True))
        self.worker.start()

    def on_row_started(self, name: str):
        row = self.find_row(name)
        if row < 0:
            return
        self.set_item(row, 3, "...", QColor("black"))
        self.set_item(row, 4, "POLLING", QColor("black"))

    def on_row_result(self, result: BatteryResult):
        row = self.find_row(result.name)
        drone = self.drone_by_name(result.name)
        if row < 0 or drone is None:
            return

        if result.status == "OK" and result.voltage is not None:
            voltage_text = f"{result.voltage:.2f} V"
            if result.voltage < drone.voltage_critical:
                color = QColor("#c62828")
                bold = True
            elif result.voltage < drone.voltage_warning:
                color = QColor("#ef6c00")
                bold = False
            else:
                color = QColor("#2e7d32")
                bold = False
            self.set_item(row, 3, voltage_text, color, bold=bold)
            self.set_item(row, 4, "OK", color)
            return

        status = result.status
        if result.detail:
            status = f"{status}: {result.detail}"
        self.set_item(row, 3, "OFFLINE", QColor("gray"), italic=True)
        self.set_item(row, 4, status, QColor("gray"), italic=True)

    def on_sweep_done(self, elapsed_s: float):
        now = datetime.now().strftime("%H:%M:%S")
        minutes = int(elapsed_s // 60)
        seconds = int(round(elapsed_s % 60))
        self.status_label.setText(f"Last sweep: {now}  ({minutes}m {seconds}s elapsed)")
        self.refresh_button.setEnabled(True)

    def on_auto_refresh_changed(self):
        if self.auto_refresh.isChecked():
            self.timer.start(self.interval_spin.value() * 1000)
        else:
            self.timer.stop()

    def on_interval_changed(self):
        if self.auto_refresh.isChecked():
            self.timer.start(self.interval_spin.value() * 1000)

    def set_item(
        self,
        row: int,
        column: int,
        text: str,
        color: QColor,
        *,
        bold: bool = False,
        italic: bool = False,
    ):
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(text)
        item.setForeground(color)
        font = QFont()
        font.setBold(bold)
        font.setItalic(italic)
        item.setFont(font)
        if column == 3:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Crazyflie fleet battery voltage.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Crazyswarm2 YAML file")
    parser.add_argument("--all", action="store_true", help="Poll disabled drones too")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cflib.crtp.init_drivers()

    app = QApplication(sys.argv)
    window = FleetBatteryWindow(args.config, args.all)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

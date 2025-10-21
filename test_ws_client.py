"""Simple WebSocket client to test the /ws endpoint."""
import asyncio
import json
import websockets


async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")

            # Receive 3 messages
            for i in range(3):
                message = await websocket.recv()
                data = json.loads(message)

                if data["type"] == "state":
                    payload = data["payload"]
                    print(f"\n--- Message {i+1} ---")
                    print(f"Particle count: {len(payload['particles'])}")
                    if payload['particles']:
                        p = payload['particles'][0]
                        print(f"First particle: x={p['x']:.2f}, y={p['y']:.2f}, species={p['species']}")
                    print(f"Config: species_count={payload['config']['species_count']}, "
                          f"particle_count={payload['config']['particle_count']}")
                else:
                    print(f"Unexpected message type: {data['type']}")
                    print(f"Message: {data}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())

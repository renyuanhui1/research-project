import asyncio
import projectairsim
from projectairsim import Drone, World

SCENE_CONFIG = "scene_px4_sitl_wsl2.jsonc"


async def main():
    client = projectairsim.ProjectAirSimClient(address="172.21.192.1")

    try:
        client.connect()
        world = World(client, SCENE_CONFIG, delay_after_load_sec=2)

        print("Scene loaded. Waiting for PX4 to connect...")
        print("Start PX4: export PX4_SIM_HOST_ADDR=172.21.192.1 && cd ~/PX4-Autopilot && make px4_sitl none_iris")
        print("Press Ctrl+C to disconnect.")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())

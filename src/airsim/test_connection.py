from projectairsim import ProjectAirSimClient, World

SCENE_CONFIG = "scene_hexarotor_drone.jsonc"

client = ProjectAirSimClient()
client.connect()

world = World(client, SCENE_CONFIG, delay_after_load_sec=2)

print(f"Loaded {SCENE_CONFIG} successfully.")
print("Drone is ready. Press Ctrl+C to disconnect.")

try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

client.disconnect()
print("Disconnected.")

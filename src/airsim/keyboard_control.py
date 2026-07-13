import asyncio
import curses
import projectairsim
from projectairsim import Drone, World

SCENE_CONFIG = "scene_drone_sensors.jsonc"

SPEED = 5.0       # m/s
YAW_SPEED = 20.0  # degrees/s
DURATION = 0.1    # seconds


async def run_keyboard_control(drone, stdscr):
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.keypad(True)

    drone.enable_api_control()
    drone.arm()

    stdscr.addstr(0, 0, "Taking off...")
    stdscr.refresh()
    task = await drone.takeoff_async()
    await task

    stdscr.clear()
    stdscr.addstr(0, 0, "=== Keyboard Control ===")
    stdscr.addstr(1, 0, "W/S: Forward/Backward")
    stdscr.addstr(2, 0, "A/D: Left/Right")
    stdscr.addstr(3, 0, "Up/Down: Altitude")
    stdscr.addstr(4, 0, "Left/Right: Yaw")
    stdscr.addstr(5, 0, "L: Land    Q: Quit")
    stdscr.refresh()

    keep_running = True
    while keep_running:
        key = stdscr.getch()

        vx, vy, vz, yaw_rate = 0, 0, 0, 0

        if key == ord('w'):
            vx = SPEED
        elif key == ord('s'):
            vx = -SPEED
        elif key == ord('a'):
            vy = -SPEED
        elif key == ord('d'):
            vy = SPEED
        elif key == curses.KEY_UP:
            vz = -SPEED
        elif key == curses.KEY_DOWN:
            vz = SPEED
        elif key == curses.KEY_LEFT:
            yaw_rate = -YAW_SPEED
        elif key == curses.KEY_RIGHT:
            yaw_rate = YAW_SPEED
        elif key == ord('l'):
            stdscr.addstr(7, 0, "Landing...          ")
            stdscr.refresh()
            task = await drone.land_async()
            await task
            keep_running = False
        elif key == ord('q'):
            keep_running = False

        if keep_running and (vx != 0 or vy != 0 or vz != 0):
            await drone.move_by_velocity_body_frame_async(vx, vy, vz, DURATION)
        if keep_running and yaw_rate != 0:
            await drone.rotate_by_yaw_rate_async(yaw_rate, DURATION)

        await asyncio.sleep(0.02)

    drone.disarm()
    drone.disable_api_control()


async def main():
    client = projectairsim.ProjectAirSimClient(address="172.21.192.1")
    stdscr = None
    try:
        client.connect()
        world = World(client, SCENE_CONFIG, delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")

        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.nodelay(True)

        await run_keyboard_control(drone, stdscr)

    except Exception as e:
        if stdscr:
            curses.endwin()
        print(f"Error: {e}")
    finally:
        if stdscr:
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()
        client.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())

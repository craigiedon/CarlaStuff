import carla

from carlaUtils import set_rendering

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
client.load_world("Town01")
world = client.get_world()
set_rendering(world, client, False)

while True:
    # print("Waiting for tick.........")
    try:
        world.tick(0.001)
    except Exception as err:
        print("Couldnt get the tick!")
        print(err)
    # print("GOT IT")

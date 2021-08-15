
import cv2
import numpy as np
import random

import carla


IM_WIDTH=640
IM_HEIGHT=480

def main():

    actors=[]
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    blueprint_library = world.get_blueprint_library()

    vehicle_num=1
    vehicles=[]

    print("add vehicle")

    try:
        for n in range(0,vehicle_num):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))
            vehicle_bp.set_attribute('role_name', 'autopilot')
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            #print(spawn_point)
            vehicle=world.spawn_actor(vehicle_bp, spawn_point)
            #vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            actors.append(vehicle)
            print(str(vehicle_bp) + "@" + str(spawn_point))
            world.project_point(spawn_point.location, carla.Vector3D(x=0,y=0,z=0),300)
            #waypoint = map.get_waypoint(spawn_point.location)

        ###sensor     
        print("add camera")
        
        cam_bp= blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f'{IM_WIDTH}')
        cam_bp.set_attribute("image_size_y", f'{IM_HEIGHT}')
        cam_bp.set_attribute("fov", "110")
        cam_bp.set_attribute('sensor_tick', '2.0')
        print(str(cam_bp))
        
        myvehicle=vehicles.pop()
        #myvehicle.apply_control(carla.VehicleControl(throttle=0.1,steer=0.0))
        sensor= world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=myvehicle)
        print(str(sensor))
        sensor.listen(lambda data: process_img(data))
        actors.append(sensor)

        print ("attach to:" + str(myvehicle))



        while True:
            #cur=myvehicle.get_location()
            #toLocation=cur
            #toLocation.x+=40
            #vehicle.set_location(toLocation)
            world.tick()

    except BaseException as err:
        print(err)
        pass

    finally:
        for actor in actors:
            actor.destroy()
            print("Terminated")

    # Tick the server

def process_img(img):
    print("process...")
    #img.save_to_disk('./sensor/%06d.png' %img.frame)
    i=np.array(img.raw_data)
    #print(i[0:-1])
    reshape_img=i.reshape(IM_HEIGHT,IM_WIDTH,4)
    rgb_reshape_img=reshape_img[:,:,:3]
    cv2.imshow("camera",rgb_reshape_img)
    cv2.waitKey(1)
    #print(rgb_reshape_img[0:3])
    #print(dir(img))
    #cv2.imshow("",rgb_reshape_img)
    #plt.show()
    #IMG_DATA=rgb_reshape_img


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')


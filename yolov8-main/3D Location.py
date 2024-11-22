import cv2
import pyrealsense2 as rs
import time
import numpy as np
import math
from ultralytics import YOLO


model = YOLO("$.pt")


# cap = cv2.VideoCapture(1)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)


pipeline.start(config)
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)


def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()


    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''


    # with open('./intrinsics.json', 'w') as fp:
    # json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))
    color_image = np.asanyarray(color_frame.get_data())


    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)
    # print ('depth: ',dis)
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate



fps = 0
frame_count = 0
start_time = time.time()

try:
    while True:

        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()

        if not depth_image.any() or not color_image.any():
            continue


        time1 = time.time()


        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))


        results = model.predict(color_image, conf=0.5)
        annotated_frame = results[0].plot()
        detected_boxes = results[0].boxes.xyxy

        for i, box in enumerate(detected_boxes):
            x1, y1, x2, y2 = map(int, box)


            xrange = max(1, math.ceil(abs((x1 - x2) / 30)))
            yrange = max(1, math.ceil(abs((y1 - y2) / 30)))
            # xrange = 1
            # yrange = 1

            point_cloud_data = []


            for x_position in range(x1, x2, xrange):
                for y_position in range(y1, y2, yrange):
                    depth_pixel = [x_position, y_position]
                    dis, camera_coordinate = get_3d_camera_coordinate(depth_pixel, aligned_depth_frame,
                                                                      depth_intrin)
                    point_cloud_data.append(f"{camera_coordinate} ")


            with open("point_cloud_data.txt", "a") as file:
                file.write(f"\nTime: {time.time()}\n")
                file.write(" ".join(point_cloud_data))


            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)
            dis, camera_coordinate = get_3d_camera_coordinate([ux, uy], aligned_depth_frame,
                                                              depth_intrin)
            formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"

            cv2.circle(annotated_frame, (ux, uy), 4, (255, 255, 255), 5)
            cv2.putText(annotated_frame, formatted_camera_coordinate, (ux + 20, uy + 10), 0, 1,
                        [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)



        cv2.imshow('YOLOv8 RealSense', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
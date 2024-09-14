import os
import time
import cv2
from ultralytics import YOLO
from common.camera import *
from common.model import *
from common.utils import Timer, evaluate
from common.generators import UnchunkedGenerator
from bvh_skeleton import openpose_skeleton,h36m_skeleton
import warnings
warnings.filterwarnings("error")

# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()

def get_video_points_data(video_path):
    # 单人视频检测
    # 窗口设置 cv2.WINDOW_NORMAL 可调节大小
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    results = model(video_path,stream=True,device='cpu')
    all_points = np.zeros((1,17,3))
    for result in results:
        keypoints = result.keypoints  # Keypoints object for pose outputs
        res_plotted = result.plot(boxes=False)
        cv2.imshow("result",res_plotted)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if keypoints.data.shape[1] == 17:
            # print(17, keypoints.data.size(), keypoints.data.shape, keypoints.data)
            all_points = np.concatenate((all_points,keypoints.data),axis=0)
        else:
            print(17, keypoints.data.size(), keypoints.data.shape, keypoints.data)
    print(18,all_points.shape,all_points)
    return all_points

def get_3d_points(points_2d):
    # 第一步：转换2D关键点
    keypoints = points_2d  # (N, 17, 2)
    # 第二步：将2D关键点转换为3D关键点
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)

    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3],)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('-------------- load data spends {:.2f} seconds'.format(ckpt))

    # load trained model
    chk_filename = './pretrained_h36m_detectron_coco.bin'
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint['model_pos'])

    ckpt, time2 = ckpt_time(time1)
    print('-------------- load 3D model spends {:.2f} seconds'.format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    print('Rendering...')
    input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)

    # save 3D joint points 保存三维关节点
    np.save('test_3d_output.npy', prediction, allow_pickle=True)

    # 第三步：将预测的三维点从相机坐标系转换到世界坐标系
    # （1）第一种转换方法
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)
    # We don't have the trajectory, but at least we can rebase the height将预测的三维点的Z值减去预测的三维点中Z的最小值，得到正向的Z值
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    # 第四步：将3D关键点输出并将预测的3D点转换为bvh骨骼
    # 将三维预测点输出
    write_3d_point('3dpoint.txt', prediction)

    # 将预测的三维骨骼点转换为bvh骨骼
    prediction_copy = np.copy(prediction)

    write_standard_bvh("./", prediction_copy)  # 转为标准bvh骨骼

    anim_output = {'Reconstruction': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))
    print('total spend {:2f} second'.format(ckpt))


# 将预测3d关键点输出到outputs/outputvideo/alpha_pose_视频名/3dpoint下
def write_3d_point(outvideopath,prediction3dpoint):
    '''
    :param prediction3dpoint: 预测的三维字典
    :param outfilepath: 输出的三维点的文件
    :return:
    '''

    frameNum = 1

    for frame in prediction3dpoint:
        file = open(d3_point_file_path, 'w')
        frameNum += 1
        for point3d in frame:
            #（2）未转换任何坐标系的输出，Z轴向上，X向右，Y向前
            str = '{},{},{}\n'.format(point3d[0],point3d[1],point3d[2])
            file.write(str)
        file.close()

# 将3dpoint转换为标准的bvh格式并输出到outputs/outputvideo/alpha_pose_视频名/bvh下
def write_standard_bvh(outbvhfilepath,prediction3dpoint):
    '''
    :param outbvhfilepath: 输出bvh动作文件路径
    :param prediction3dpoint: 预测的三维关节点
    :return:
    '''

    # 将预测的点放大100倍
    for frame in prediction3dpoint:
        for point3d in frame:
            point3d[0] *= 100
            point3d[1] *= 100
            point3d[2] *= 100

    bvhfileName = bvhfile_path
    human36m_skeleton = h36m_skeleton.H36mSkeleton()
    human36m_skeleton.poses2bvh(prediction3dpoint,output_file=bvhfileName)

if __name__ == "__main__":
    video_path = r"E:\摔倒项目\ceshi.mp4"
    d3_point_file_path = f"./{os.path.basename(video_path).split('.')[0]}.txt"
    bvhfile_path = f"./{os.path.basename(video_path).split('.')[0]}.bvh"
    # Load the YOLOv8 model
    model = YOLO('../yolov8n-pose.pt')

    metadata = {'layout_name': 'coco', 'num_joints': 17,
                'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

    all_points = get_video_points_data(video_path)
    get_3d_points(all_points)

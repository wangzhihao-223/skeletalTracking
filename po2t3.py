import os
import time

import cv2
import numpy as np
from common.camera import *
from common.model import *
from common.utils import Timer, evaluate
from common.generators import UnchunkedGenerator
from common.arguments import parse_args
from bvh_skeleton import openpose_skeleton,h36m_skeleton


metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


def get_detector_2d(detector_name):
    def get_alpha_pose():
        from .joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
        return alpha_pose

    def get_hr_pose():
        from .joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
        return hr_pose

    detector_map = {
        'alpha_pose': get_alpha_pose,
        'hr_pose': get_hr_pose,
        # 'open_pose': open_pose
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]

def main(args):
    # 第一步：检测2D关键点

    keypoints = np.array(args.input_npz)  # (N, 17, 2)

    # 第二步：将2D关键点转换为3D关键点
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # normlization keypoints  Suppose using the camera parameter
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)

    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
                              dense=args.dense)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('-------------- load data spends {:.2f} seconds'.format(ckpt))

    # load trained model
    chk_filename = args.evaluate
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
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
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

    # （2）第二种转换方法
    # subject = 'S1'
    # cam_id = '55011271'
    # cam_params = load_camera_params('./camera/cameras.h5')[subject][cam_id]
    # R = cam_params['R']
    # T = 0
    # azimuth = cam_params['azimuth']
    #
    # prediction = camera2world(pose=prediction, R=R, T=T)
    # prediction[:, :, 2] -= np.min(prediction[:, :, 2])  # rebase the height

    # 第四步：将3D关键点输出并将预测的3D点转换为bvh骨骼
    # 将三维预测点输出
    write_3d_point(args.viz_output,prediction)

    # 将预测的三维骨骼点转换为bvh骨骼
    prediction_copy = np.copy(prediction)

    write_standard_bvh(args.viz_output,prediction_copy) #转为标准bvh骨骼
    # write_smartbody_bvh(args.viz_output,prediction_copy) #转为SmartBody所需的bvh骨骼

    anim_output = {'Reconstruction': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

    # if not args.viz_output:
    #     args.viz_output = 'alpha_result.mp4'
    #
    # # 第五步：生成输出视频
    # from common.visualization import render_animation
    # render_animation(input_keypoints, anim_output,
    #                  Skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
    #                  limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
    #                  input_video_path=args.viz_video, viewport=(1000, 1002),
    #                  input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    print('total spend {:2f} second'.format(ckpt))


def inference_video(video_path, detector_2d):
    """
    Do image -> 2d points -> 3d points to video.
    :param detector_2d: used 2d joints detector. Can be {alpha_pose, hr_pose}
    :param video_path: relative to outputs
    :return: None
    """
    from ultralytics import YOLO
    model = YOLO('../yolov8n-pose.pt')
    img = cv2.imread(r'../persons3.jpg')
    results = model(img)  # predict on an image
    res_xy = results[0].keypoints.xy

    args = parse_args()
    # args.detector_2d = detector_2d
    # args.viz_video = video_path
    args.input_npz = res_xy

    args.evaluate = 'pretrained_h36m_detectron_coco.bin'

    with Timer(video_path):
        main(args)

# 将预测3d关键点输出到outputs/outputvideo/alpha_pose_视频名/3dpoint下
def write_3d_point(outvideopath,prediction3dpoint):
    '''
    :param prediction3dpoint: 预测的三维字典
    :param outfilepath: 输出的三维点的文件
    :return:
    '''

    frameNum = 1

    for frame in prediction3dpoint:
        # outfileDirectory = os.path.join(dir_name,video_name,"3dpoint");
        # if not os.path.exists(outfileDirectory):
        #     os.makedirs(outfileDirectory)
        # outfilename = os.path.join(dir_name,video_name,"3dpoint","3dpoint{}.txt".format(frameNum))
        # file = open(outfilename, 'w')
        file = open('ceshi01.txt', 'w')
        frameNum += 1
        for point3d in frame:
            # （1）转换成SmartBody和Meshlab的坐标系，Y轴向上，X向右，Z轴向前
            # X = point3d[0]
            # Y = point3d[1]
            # Z = point3d[2]
            #
            # X_1 = -X
            # Y_1 = Z
            # Z_1 = Y
            # str = '{},{},{}\n'.format(X_1, Y_1, Z_1)

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

            # 交换Y和Z的坐标
            #X = point3d[0]
            #Y = point3d[1]
            #Z = point3d[2]

            #point3d[0] = -X
            #point3d[1] = Z
            #point3d[2] = Y


    bvhfileName = "qq.bvh"
    human36m_skeleton = h36m_skeleton.H36mSkeleton()
    human36m_skeleton.poses2bvh(prediction3dpoint,output_file=bvhfileName)


if __name__ == '__main__':

    inference_video(r'../persons3.jpg', 'alpha_pose')

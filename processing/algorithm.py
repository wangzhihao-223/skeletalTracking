
    # 因该系统需要演示9种算法，所以将此算法独立出来
    import cv2
    import numpy as np



    # 预处理——去噪类
    class algo_filter():
        def bilatFilter(frame):
            # 从摄像头读取视频，参数 0 表示默认摄像头
            bilateral_frame = cv2.bilateralFilter(frame, 9, 75, 75)
            return bilateral_frame


        def gaussFilter(frame):
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            return blurred_frame


        def meanFilter(frame):
            averaged_frame = cv2.blur(frame, (5, 5))
            return averaged_frame


        def medFilter(frame):
            # 中值滤波去噪
            median_frame = cv2.medianBlur(frame, 5)
            return median_frame


    # 预处理——灰度图处理类
    class algo_gray():
        def floatGray(frame):
            adjustment = 1.5
            # Convert the frame to grayscale
            #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert pixel values to float32
            frame_float32 = frame.astype(np.float32)
            # Apply brightness adjustment to each pixel
            adjusted_frame_float32 = frame_float32 * adjustment
            # Limit pixel values to the range 0-255
            adjusted_frame_float32[adjusted_frame_float32 < 0] = 0
            adjusted_frame_float32[adjusted_frame_float32 > 255] = 255
            # Convert pixel values back to uint8 type
            adjusted_frame = adjusted_frame_float32.astype(np.uint8)
            return adjusted_frame


        

        def intGray(frame):
            # 整型灰度处理
            #grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert pixel values to integers
            frame_int = frame.astype(int)
            # Example processing: Invert the image
            inverted_frame_int = 255 - frame_int
            # Convert pixel values back to uint8 type
            inverted_frame = inverted_frame_int.astype(np.uint8)
            return inverted_frame


        def moveGray(frame):
            factor = 1.5
            # 将图像数据类型转换为int16
            frame_int16 = frame.astype(np.int16)
            # 应用对比度调整
            adjusted_frame_int16 = frame_int16 * factor
            # 将超出范围的像素值限制在0-255之间
            adjusted_frame_int16[adjusted_frame_int16 < 0] = 0
            adjusted_frame_int16[adjusted_frame_int16 > 255] = 255
            # 将图像数据类型转换回uint8
            adjusted_frame = adjusted_frame_int16.astype(np.uint8)
            return adjusted_frame


    # 追踪算法的实现类
    class algo_tracking():
        # 质心追踪算法
        def centroid(frame): # 输入一个帧
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 使用阈值分割（参数可以根据实际情况调整）
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            # 寻找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 遍历每个轮廓
            for contour in contours:
                # 计算轮廓的质心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 在图像上标注质心
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            return frame

    # 用于切换算法的类
    class algo_switch: # 用于切换算法的类
        # 定义算法状态字典，用以保存记录算法是否处于开启状态
        algorithm_states = {
            "gaussFilter": False,  # 预处理——高斯去噪
            "meanFilter": False,   # 预处理——均值去噪
            "medFilter": False,   # 预处理——中值去噪
            "bilatFilter":False,    # 预处理——双边滤波去噪
            "floatGray": False,     # 灰度处理——浮点操作
            "intGray": False,       # 灰度处理——整型
            "moveGray": False,      # 灰度处理——移位
            "centroid": False,      # 质心追踪算法
            # 添加更多的算法...
        }

        # 函数映射字典
        # 你可以在这里添加其他函数
        algorithm_functions = {
            "gaussFilter": algo_filter.gaussFilter,
            "meanFilter": algo_filter.meanFilter,
            "medFilter": algo_filter.medFilter,  
            "bilatFilter": algo_filter.bilatFilter,
            "floatGray": algo_gray.floatGray,
            "intGray": algo_gray.intGray,
            "moveGray": algo_gray.moveGray,
            "centroid": algo_tracking.centroid,
        }

        # 查询一次算法状态字典，看当前是否有算法激活？
        def check_active_algo(frame):
            for algorithm, state in algo_switch.algorithm_states.items():
                # 如果有算法处于激活状态，使用该函数处理图像
                if state:
                    print("check_active_algo已处理")
                    return algo_switch.algorithm_functions[algorithm](frame)
                # 如果没有激活，不操作
            return frame

        # 切换算法状态
        def switch_algorithm(algorithm_name):
            # 遍历状态字典，将其他所有算法状态置为关闭
            for algorithm, state in algo_switch.algorithm_states.items():
                if state: # 如果有任何算法处于激活状态
                    algo_switch.algorithm_states[algorithm] = False # 将所有算法置于关闭状态
                    return # 直接结束函数
            # 否则将输入算法置于激活状态 
            algo_switch.algorithm_states[algorithm_name] = True
            print(f"{algorithm_name} algorithm is already activated,now the dic is{algo_switch.algorithm_states}")

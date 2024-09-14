
import time
import threading
from core.camera import camera

class SaveVideoThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.frame_interval = 1 / 24
        self.start_time = time.perf_counter()
        self.success = False
        self.lock = threading.Lock()  # 创建锁对象

    def run(self):
        print("现在开始录制视频！")
        frame_time = self.start_time
        
        while True:
            with self.lock:
                # camera.tmp_im2 是捕获的视频帧
                camera.result.write(camera.tmp_im)
            time.sleep(self.frame_interval)
            frame_time += self.frame_interval
            
            # 检查是否录制结束
            if frame_time >= self.start_time + 5:  # 后5秒
                self.success = True
                camera.result.release()
                break

        if self.success:
            print("视频保存成功！")
        else:
            print("视频保存失败！")

def video_record():
    # 创建线程并启动
    save_video_thread = SaveVideoThread()
    save_video_thread.start()

from PIL import Image
import io
import matplotlib.pyplot as plt
import os
class Visualize():
    def __init__(self, save_name, test_id) -> None:
        self.history = {
            'acc': [],
            'acc_traci': [],
            'vel': [],
            'steps': [],
            'speed_lim': [],
            'rewards':[],
            'desired_speed': [],
            'desired_gap': [],
            'leader_speed': [],
            'leader_gap': [],
            'current_gap': [],
            'jerk': [],
            'headway': [],
            'ttc': []
        }
        snap_path = os.path.join(save_name, 'snaps/')
        if not os.path.exists(snap_path):
            os.makedirs(snap_path)
        self.save_name = save_name
        self.test_id = test_id
        # self.test_id = test_id

    def get_concat_h(self,im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(self, im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    def set_history(self, data):
        self.history = data

    def get_concat_h_resize(self, im1, im2, resample=Image.BICUBIC, resize_big_image=True):
        if im1.height == im2.height:
            _im1 = im1
            _im2 = im2
        elif (((im1.height > im2.height) and resize_big_image) or
            ((im1.height < im2.height) and not resize_big_image)):
            _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
            _im2 = im2
        else:
            _im1 = im1
            _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
        dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
        dst.paste(_im1, (0, 0))
        dst.paste(_im2, (_im1.width, 0))
        return dst

    def get_concat_v_resize(self, im1, im2, resample=Image.BICUBIC, resize_big_image=True):
        if im1.width == im2.width:
            _im1 = im1
            _im2 = im2
        elif (((im1.width > im2.width) and resize_big_image) or
            ((im1.width < im2.width) and not resize_big_image)):
            _im1 = im1.resize((im2.width, int(im1.height * im2.width / im1.width)), resample=resample)
            _im2 = im2
        else:
            _im1 = im1
            _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width)), resample=resample)
        dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
        dst.paste(_im1, (0, 0))
        dst.paste(_im2, (0, _im1.height))
        return dst
    
    def compute_jerk(self, acceleration_list):
        diff_list = []
        ini_list = acceleration_list
        for x1, y1 in zip(ini_list[0::], ini_list[1::]):
            diff_list.append((y1-x1)/0.1)
        return diff_list
    
    def overlay(self):
    
        steps = self.history['steps']
        acc_list = self.history['acc_traci']
        velocity_list = self.history['vel']
        speed_lim = self.history['speed_lim']
        reward = self.history['rewards']
        desired_gap_list = self.history['desired_gap']
        desired_speed_list = self.history['desired_speed']
        current_gap_list = self.history['current_gap']
        # jerk_list = self.history['jerk']

        jerk_list = self.compute_jerk(acc_list)
        # fig, ax = plt.subplots(1,1)
        # plt.show()
        over_path = f"{self.save_name}/overlayed"
        
        if not os.path.exists(over_path):
            os.makedirs(over_path)
        for step in steps:
            fig = plt.figure()
            ax1 = fig.add_subplot(511)
            ax2 = fig.add_subplot(512)
            ax3 = fig.add_subplot(513)
            ax4 = fig.add_subplot(514)
            ax5 = fig.add_subplot(515)
            if os.path.exists(f'{self.save_name}/snaps/{step}.png'):
                img = Image.open(f'{self.save_name}/snaps/{step}.png')
                img_buf = io.BytesIO()
                
                ax1.set_ylim(-5.0,5.0)
                ax1.set_ylabel('Acceleration (m/s^2)')
                # ax1.plot(steps[0:step], acc_list[0:step])
                off = int(step*10)
                ax1.plot(range(len(acc_list[0:off])), acc_list[0:off])
                # plt.savefig(img_buf, format='png')
                
                # img2 = Image.open(img_buf)
                
                ax2.set_ylim(0.0,60.0)
                ax2.set_ylabel('Velocity (m/s)')
                ax2.plot(steps[0:step], velocity_list[0:step], label="Speed")
                ax2.plot(steps[0:step], speed_lim[0:step], label="Speed Limit")
                ax2.plot(steps[0:step], desired_speed_list[0:step], label="Desired Speed")
                ax2.legend()

                # ax3.set_ylim(0.0,1.0)
                ax3.set_ylim(-9, 1)
                ax3.set_ylabel('Reward')
                ax3.plot(steps[0:step], reward[0:step])

                ax4.set_ylim(0.0,160.0)
                ax4.set_ylabel('Gap (m)')
                ax4.plot(steps[0:step], desired_gap_list[0:step], label="Desired Gap")
                ax4.plot(steps[0:step], current_gap_list[0:step], label="Current Gap")
                ax4.legend()

                ax5.set_ylim(-30.0,30.0)
                ax5.set_ylabel('Jerk (m/s^2)')
                # ax5.plot(steps[0:step], jerk_list[0:step])
                ax5.plot(range(len(jerk_list[0:off])), jerk_list[0:off])
                # plt.savefig(img_buf2, format='png')
                # img3 = Image.open(img_buf2)
                # ax.plot(steps, acc_list,'k-', lw=2)
                # plt.savefig(f'{over_path}/{step}.png')
                fig.savefig(img_buf, format='png')
                img_buf.seek(0)
                
                # img2 = Image.frombytes('RGB',fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                img2 = Image.open(img_buf)
                # img.paste(img2, (0,0))

                # img.paste(img3, (0, 300))
                # img.save(f'{over_path}/{step}.png')
                self.get_concat_h(img, img2).save(f'{over_path}/{step}.png')
                img_buf.close()
                plt.close()


    def save(self):
        os.system(f"ffmpeg -r 10 -i {self.save_name}/overlayed/%01d.png -vcodec mpeg4 -y {self.save_name}/movie_test_{self.test_id}.mp4")
        # os.system(f"rm -rf videos/test{self.test_id}/snaps")

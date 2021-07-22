from tensorboardX import SummaryWriter
from pathlib import Path
import os

class Mylogger():
    def __init__(self, model_name):
        model_dir = Path('./models') / model_name
        if not model_dir.exists():
            run_num = 1
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                            model_dir.iterdir() if
                            str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                run_num = 1
            else:
                run_num = max(exst_run_nums) + 1
        curr_run = 'run%i' % run_num
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        os.makedirs(log_dir)
        self.logger = SummaryWriter(str(log_dir))

    def close(self):
        self.logger.close()
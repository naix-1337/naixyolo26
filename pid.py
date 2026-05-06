from simple_pid import PID


class PIDController:
    def __init__(self, kp=0.7, ki=0.0, kd=0.0, output_limits=(-40, 40)):
        self.pid_x = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=0, sample_time=None)
        self.pid_y = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=0, sample_time=None)
        self.pid_x.output_limits = output_limits
        self.pid_y.output_limits = output_limits

    def update(self, error_x, error_y):
        return self.pid_x(-error_x), self.pid_y(-error_y)

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()

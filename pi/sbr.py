import time
import threading
import math

from a_star import AStar
from lsm6 import LSM6

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

# This code was developed for a Balboa unit using 50:1 motors
# and 45:21 plastic gears, for an overall gear ratio of 111.
# Adjust the ratio below to scale various constants in the
# balancing algorithm to match your robot.
GEAR_RATIO = 111

# This constant limits the maximum motor speed.  If your gear
# ratio is lower than what we used, or if you are testing
# changes to the code, you might want to reduce it to prevent
# your robot from zooming away when things go wrong.
MOTOR_SPEED_LIMIT = 300

# This constant relates the angle to its rate of change for a
# robot that is falling from a nearly-vertical position or
# rising up to that position.  The relationship is nearly
# linear.  For example, if you have the 80mm wheels it should be
# about 140, which means that the angle in millidegrees is ~140
# times its rate of change in degrees per second; when the robot
# has fallen by 90 degrees it will be moving at about
# 90,000/140 = 642 deg/s.  See the end of Balancer.ino for one
# way to calibrate this value.
ANGLE_RATE_RATIO = 140

# The following three constants define a PID-like algorithm for
# balancing.  Each one determines how much the motors will
# respond to the corresponding variable being off from zero.
# See the code in Balance.cpp for exactly how they are used.  To
# get it balancing from scratch, start with them all at zero and
# adjust them as follows:

# ANGLE_RESPONSE determines the response to a combination of
# angle and angle_rate; the combination measures how far the
# robot is from a stable trajectory.  To test this, use your
# hand to flick the robot up from a resting position.  With a
# value that is too low, it won't stop itself in time; a high
# value will cause it to slam back into the ground or oscillate
# wildly back and forth.  When ANGLE_RESPONSE is adjusted
# properly, the robot will move just enough to stop itself.
# However, after stopping itself, it will be moving and keep
# moving in the same direction, usually driving faster and
# faster until it reaches its maximum motor speed and falls
# over.  That's where the next constants come in.
ANGLE_RESPONSE = 12

# DISTANCE_RESPONSE determines how much the robot resists being
# moved away from its starting point.  Counterintuitively, this
# constant is positive: to move forwards, the robot actually has
# to first roll its wheels backwards, so that it can *fall*
# forwards.  When this constant is adjusted properly, the robot
# will no longer zoom off in one direction, but it will drive
# back and forth a few times before falling down.
DISTANCE_RESPONSE = 90

# DISTANCE_DIFF_RESPONSE determines the response to differences
# between the left and right motors, preventing undesired
# rotation due to differences in the motors and gearing.  Unlike
# DISTANCE_REPONSE, it should be negative: if the left motor is
# lagging, we need to increase its speed and decrease the speed
# of the right motor.  If this constant is too small, the robot
# will spin left and right as it rocks back and forth; if it is
# too large it will become unstable.
DISTANCE_DIFF_RESPONSE = -50

# SPEED_RESPONSE supresses the large back-and-forth oscillations
# caused by DISTANCE_RESPONSE.  Increase this until these
# oscillations die down after a few cycles; but if you increase
# it too much it will tend to shudder or vibrate wildly.
SPEED_RESPONSE = 3300

# The balancing code is all based on a 100 Hz update rate; if
# you change this, you will have to adjust many other things.
UPDATE_TIME = 0.01

# The sensitivity of the gyroscope.
GYROSCOPE_SENSITIVITY = 29

# Take 100 measurements initially to calibrate the gyro.
CALIBRATION_ITERATIONS = 200

class Balancer:
  def __init__(self):
    self.a_star = AStar()
    self.imu = LSM6()

    self.g_y_zero = 0
    self.angle = 0 # degrees
    self.angle_rate = 0 # degrees/s
    self.distance_left = 0
    self.speed_left = 0
    self.drive_left = 0
    self.last_counts_left = 0
    self.distance_right = 0
    self.speed_right = 0
    self.drive_right = 0
    self.last_counts_right = 0

    self.motor_speed = 0
    self.motor_speed_sim = 0 # the sim take different values than that of the motor.
                             # we will just map the values to the correct range.

    self.balancing = False
    self.calibrated = False
    self.running = False
    self.next_update = 0
    self.update_thread = None
    self.pitch = 0
    self.pitch_euler = 0
    self.roll = 0
    self.roll_euler = 0
    self.pitchAcc = 0

  def setup(self):
    self.imu.enable()
    time.sleep(1) # wait for IMU readings to stabilize

    self.roll = self.imu.g.x
    self.pitch = self.imu.g.y
    for _ in range(CALIBRATION_ITERATIONS):
      self.imu.read()
      self.getPitchEuler()
      time.sleep(UPDATE_TIME)
    self.calibrated = True

  def start(self):
    if self.calibrated:
      if not self.running:
        self.running = True
        self.next_update = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
    else:
      raise RuntimeError("IMU not enabled/calibrated; can't start balancer")

  def stop(self):
    if self.running:
      self.a_star.motors(0, 0)
      self.running = False
      self.update_thread.join()

  def make(self):
    self.setup()
    self.start()

  def stand_up(self):
    if self.calibrated:
      sign = 1

      if self.imu.a.z < 0:
        sign = -1

      self.stop()
      self.reset()
      self.imu.read()
      self.a_star.motors(-sign*MOTOR_SPEED_LIMIT, -sign*MOTOR_SPEED_LIMIT)
      time.sleep(0.4)
      self.a_star.motors(sign*MOTOR_SPEED_LIMIT, sign*MOTOR_SPEED_LIMIT)

      for _ in range(40):
        time.sleep(UPDATE_TIME)
        self.update_sensors()
        print(self.angle)
        if abs(self.angle) < 60:
          break

      self.motor_speed = sign*MOTOR_SPEED_LIMIT
      self.reset_encoders()
      self.start()
    else:
      raise RuntimeError("IMU not enabled/calibrated; can't stand up")


  '''def update_loop(self):
    while self.running:
      self.update_sensors()
      self.do_drive_ticks()

      print("gyro = ", self.imu.g.y)

      if self.imu.a.x < 2000:
        # If X acceleration is low, the robot is probably close to horizontal.
        self.reset()
        self.balancing = False
      else:
        self.balance()
        self.balancing = True

      # Perform the balance updates at 100 Hz.
      self.next_update += UPDATE_TIME
      now = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
      time.sleep(max(self.next_update - now, 0))

    # stop() has been called and the loop has exited. Stop the motors.
    self.a_star.motors(0, 0)'''

  def update_loop(self):
    if self.running:
      self.update_sensors()
      self.do_drive_ticks()


      # Perform the balance updates at 100 Hz.
      self.next_update += UPDATE_TIME
      now = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
      time.sleep(max(self.next_update - now, 0))

    # stop() has been called and the loop has exited. Stop the motors.
    self.a_star.motors(0, 0)


  def update_sensors(self):
    self.imu.read()
    self.integrate_gyro()
    self.integrate_encoders()

  def integrate_gyro(self):
    # Convert from full-scale 1000 deg/s to deg/s.
    self.angle_rate = (self.imu.g.y - self.g_y_zero) * 35 / 1000

    self.angle += self.angle_rate * UPDATE_TIME

  def integrate_encoders(self):
    (counts_left, counts_right) = self.a_star.read_encoders()

    self.speed_left = subtract_16_bit(counts_left, self.last_counts_left)
    self.distance_left += self.speed_left
    self.last_counts_left = counts_left

    self.speed_right = subtract_16_bit(counts_right, self.last_counts_right)
    self.distance_right += self.speed_right
    self.last_counts_right = counts_right

  def drive(self, left_speed, right_speed):
    self.drive_left = left_speed
    self.drive_right = right_speed

  def do_drive_ticks(self):
    self.distance_left -= self.drive_left
    self.distance_right -= self.drive_right
    self.speed_left -= self.drive_left
    self.speed_right -= self.drive_right

  def reset(self):
    self.motor_speed = 0
    self.reset_encoders()
    self.a_star.motors(0, 0)

    return self.imu.g.y

    '''if abs(self.angle_rate) < 2:
      # It's really calm, so assume the robot is resting at 110 degrees from vertical.
      if self.imu.a.z > 0:
        self.angle = 110
      else:
        self.angle = -110'''

  def step(self, action):
    if self.running:
        # let dv be the the max change in velocity that we can assign at a given time.
        dv = 0.1
        # scale dv to a range of actions.
        deltav = [-10.*dv, -5.*dv, -2.*dv, -0.1*dv, 0, 0.1*dv, 2.*dv, 5.*dv, 10.*dv][action]
        # vt is the current motor speed of the motor and clamp range to max of motorSpeed.
        vt = clamp(self.motor_speed + deltav, -51, 51)
        self.motor_speed_sim = vt


        # CATION: NOT SURE IF A_STAR.MOTORS(..., ...) TAKES IN A CHANGE IN SPEED OR THE CURRENT SPEED.
        # CATION: NOT SURE IF A_STAR.MOTORS(..., ...) TAKES IN A CHANGE IN SPEED OR THE CURRENT SPEED.
        # CATION: NOT SURE IF A_STAR.MOTORS(..., ...) TAKES IN A CHANGE IN SPEED OR THE CURRENT SPEED.
        # CATION: NOT SURE IF A_STAR.MOTORS(..., ...) TAKES IN A CHANGE IN SPEED OR THE CURRENT SPEED.

        # map the values from the range of +/-51 to +/-300.
        self.motor_speed = map(self.motor_speed_sim, -51, 51, -MOTOR_SPEED_LIMIT, MOTOR_SPEED_LIMIT)

        # take an action.
        self.a_star.motors(int(self.motor_speed), int(self.motor_speed))

        # now observe the results.
        obs = self.getObservations()

        # return our observations.
        return obs

  def update100Hz(self):
    # perform the update at  100Hz.
    self.next_update += UPDATE_TIME
    now = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    time.sleep(max(self.next_update - now, 0))

  def getObservations(self):
    # update the sensors.
    self.update_sensors()

    # get the pitch of the gyro
    pitch = self.getPitchEuler()

    # get the angular speed.
    angular_speed = self.getAngularSpeed()

    # get the left and right encoder counts.
    #(counts_left, counts_right) = self.a_star.read_encoders()

    # return the gyro pitch, the encoder left count, encoder right count, angular speed, and speed.
    return pitch, angular_speed, self.motor_speed_sim # should fix to return the motor speed instead of None.

  def getPitch(self):
    def complementaryFilter():
      # function combines the accelerometer and the gyroscope for a more accurate reading of pitch.
      # get the accelerometer data.
      accelData = []
      accelData.append(self.imu.a.x)
      accelData.append(self.imu.a.y)
      accelData.append(self.imu.a.z)

      # get the gyroscope data.
      gyroData = []
      gyroData.append(self.imu.g.x)
      gyroData.append(self.imu.g.y)
      gyroData.append(self.imu.g.z)

      # Using the data from the acclerometer gives a better estimate of the gyro.
      pitchAcc = 0
      rollAcc = 0

      # calculate the angle around the angle
      self.roll += gyroData[0] / GYROSCOPE_SENSITIVITY * UPDATE_TIME
      self.pitch -= gyroData[1] / GYROSCOPE_SENSITIVITY * UPDATE_TIME

      # compensate for the gyroscopic drift.
      forceMagnitudeApprox = abs(accelData[0]) + abs(accelData[1]) + abs(accelData[2])
      if (forceMagnitudeApprox > 8192 and forceMagnitudeApprox < 32768):
        rollAcc = math.atan2(accelData[1], accelData[2]) * 180 / math.pi
        self.roll = self.roll * 0.98 + rollAcc * 0.02

        pitchAcc = math.atan2(accelData[0], accelData[2]) * 180 / math.pi
        self.pitchAcc = pitchAcc
        self.pitch = self.pitch * 0.98 + pitchAcc * 0.02
    return complementaryFilter()

  def getPitchEuler(self):
    # get pitch in degrees.
    self.getPitch()

    # convert degrees to radians.
    roll_radians = self.roll * math.pi / 180
    pitch_radians = self.pitch * math.pi / 180

    # save the old values because we can use the old values to help find angular velocity.
    self.prev_pitch_euler = self.pitch_euler
    self.prev_roll_euler = self.roll_euler

    # save the values.
    # shifting the values so that zero degrees is straight up.
    self.pitch_euler = pitch_radians - (math.pi/2)
    self.roll_euler = roll_radians - (math.pi/2)

    # return the results too.
    return self.pitch_euler

  def getAngularSpeed(self):
    self.angular_speed = self.pitch_euler - self.prev_pitch_euler
    return self.angular_speed

  def reset_encoders(self):
    self.distance_left = 0
    self.distance_right = 0

  def balance(self):
    # Adjust toward angle=0 with timescale ~10s, to compensate for
    # gyro drift.  More advanced AHRS systems use the
    # accelerometer as a reference for finding the zero angle, but
    # this is a simpler technique: for a balancing robot, as long
    # as it is balancing, we know that the angle must be zero on
    # average, or we would fall over.
    self.angle *= 0.999

    # This variable measures how close we are to our basic
    # balancing goal - being on a trajectory that would cause us
    # to rise up to the vertical position with zero speed left at
    # the top.  This is similar to the fallingAngleOffset used
    # for LED feedback and a calibration procedure discussed at
    # the end of Balancer.ino.
    #
    # It is in units of degrees, like the angle variable, and
    # you can think of it as an angular estimate of how far off we
    # are from being balanced.
    rising_angle_offset = self.angle_rate * ANGLE_RATE_RATIO/1000 + self.angle

    # Combine risingAngleOffset with the distance and speed
    # variables, using the calibration constants defined in
    # Balance.h, to get our motor response.  Rather than becoming
    # the new motor speed setting, the response is an amount that
    # is added to the motor speeds, since a *change* in speed is
    # what causes the robot to tilt one way or the other.
    self.motor_speed += (
      + ANGLE_RESPONSE*1000 * rising_angle_offset
      + DISTANCE_RESPONSE * (self.distance_left + self.distance_right)
      + SPEED_RESPONSE * (self.speed_left + self.speed_right)
      ) / 100 / GEAR_RATIO

    if self.motor_speed > MOTOR_SPEED_LIMIT:
      self.motor_speed = MOTOR_SPEED_LIMIT
    if self.motor_speed < -MOTOR_SPEED_LIMIT:
      self.motor_speed = -MOTOR_SPEED_LIMIT

    # Adjust for differences in the left and right distances; this
    # will prevent the robot from rotating as it rocks back and
    # forth due to differences in the motors, and it allows the
    # robot to perform controlled turns.
    distance_diff = self.distance_left - self.distance_right
    self.a_star.motors(
      int(self.motor_speed + distance_diff * DISTANCE_DIFF_RESPONSE / 100),
      int(self.motor_speed - distance_diff * DISTANCE_DIFF_RESPONSE / 100))

def subtract_16_bit(a, b):
  diff = (a - b) & 0xFFFF
  if (diff & 0x8000):
    diff -= 0x10000
  return diff

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

# load the model.
model = DQN.load("sbr")

# create the environment.
env = Balancer()
env.make()

# initialize the first observation.
obs = env.reset()

# run the model.
for i in range(100):
    # throw into neural network to get action
    # action = NN.
    # take a random action for now.
    #action = .5

    # take the action.
    #obs = env.step(action)
    # perform 100Hz update

    #angle = env.getPitchEuler()
    #angular_speed = env.getAngularSpeed()
    #motor_speed = env.
    #print("angle:", angle, ", angular_speed:", angular_speed, ", accel:", env.pitchAcc)
    env.update_sensors()
    obs = env.getObservations()

    # stop everything if the  pitch is too low.
    if obs[0] > 0.785398 or obs[0] < -0.785398:
        env.stop()
        break

    # get the action.
    action, _states = model.predict(obs)
    # take the action.
    env.step(action)

    # delay by update time.
    env.update100Hz()


# once we are done, close the environment.
env.stop()

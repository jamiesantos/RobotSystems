import time
from sim_robot_hat.line_control import Sensor, Interpreter, Controller
from picarx.picarx_improved import Picarx
from sim_robot_hat.rossros import Bus, ConsumerProducer, Producer, Consumer, Timer, Printer, runConcurrently

picar = Picarx()

adc0, adc1, adc2 = picar.grayscale.pins

sensor = Sensor(adc0, adc1, adc2)
interpret = Interpreter()
control = Controller(picar)

sensor_bus = Bus([0,0,0], name="sensor bus")
interpret_bus = Bus(0, name="interpret bus")
termination_bus = Bus(0, name="termination bus")
     
def handle_exception(future):
    exception = future.exception()
    if exception:
        print(f"Exception in worker thread: {exception}")

def consumer_producer(bus_w): 
    print("consumer producer")
    gray_int = interpret.processing(bus_w)
    return gray_int

def consumer(turn): 
    print("consumer")
    gray_control = control.line_following(turn)

def read_sensor():
    return sensor.read()

def interpret_line(readings):
    return interpret.process(readings)

def steer(offset):
    if offset is not None:
        control.command(offset)

#def steer(offset):
#    if termination_bus.read() == 0 and offset is not None:
#        control.command(offset)


sensor_producer = Producer(
    read_sensor, # function to generate data
    sensor_bus, # output data 
    .25, # delay time 
    termination_bus, # bus to watch for termination signal 
    "Sensor Producer"
)

interpret_CP = ConsumerProducer(
    interpret_line,
    sensor_bus,
    interpret_bus,
    .25,
    termination_bus,
    "Interpretation ConsumerProducer"
)

line_follow = Consumer(
    steer,
    interpret_bus,
    .25,
    termination_bus,
    "Controller Consumer"
)

termination_timer = Timer(
    termination_bus, #output data bus
    100, # Duration 
    .01, # delay between chacking for terination time
    termination_bus, # bus to check for termination time 
    "termination timer" #name of timer
)

def listen_for_ctrl_c():
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl-C received — terminating RossROS")
        picar.set_dir_servo_angle(0)
        picar.stop()
        return 1

ctrlc_producer = Producer(
    listen_for_ctrl_c,
    termination_bus,
    0.1,
    termination_bus,
    "Ctrl-C Producer"
)

producer_consumer_list = [sensor_producer, 
                            interpret_CP,
                            line_follow,
                            ctrlc_producer,
                            termination_timer]


picar.forward(20)
runConcurrently(producer_consumer_list)
print("Car stopped safely")

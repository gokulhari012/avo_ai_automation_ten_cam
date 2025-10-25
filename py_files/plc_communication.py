# from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pymodbus.client import ModbusTcpClient as ModbusClient
import time
import json

# Configure Modbus client
# client = ModbusClient(host='192.168.0.10', port=502)  # Replace with your PLC IP and port
ip_address = ""

with open("configuration.json", "r") as file:
    json_data = json.load(file)
stepper_motor_speed = json_data["stepper_motor_speed"]

def connect():
    global client, connection
    client = ModbusClient(host=ip_address, port=502)  # Replace with your PLC IP and port
    connection = client.connect()
   
def setup(ip_address_var):
    global ip_address
    ip_address = ip_address_var
    connect()
    write(1)
    time.sleep(0.5)
    write(0)
    print("Connected to PLC")
    close()

def write(value, address=0):
    connect()
    if connection:
        # Write signal (0 or 1) to Coil (Discrete Output) at address 0
        # Replace with your actual address
        client.write_coil(address, value)  # 'unit' is the slave ID

        # Read coil status
        response = client.read_coils(address)

        if response.isError():
            print("Error reading coil")
        else:
            print(f"PLC Coil Status: {response.bits[0]}")  
            # print(f"PLC Coil response: {response}")  

        print(f"Signal {value} sent to address {address}")
    else:
        print("Failed to connect to PLC")
    close()
  
def start_stepper_motor():
    connect()
    if connection:
        client.write_register(3, 1)
        client.write_register(4, 1)
        client.write_register(0, stepper_motor_speed) # speed
        client.write_register(1, 0) # forward reverse
        client.write_register(2, 1) # speed
    close()

def stop_stepper_motor():
    connect()
    if connection:
        client.write_register(0, 0) # speed
        client.write_register(2, 1) # speed
    close()

def write_regester(value=1000,register_address=0):
    # address 0 = Speed like 1000
    # address 1 = 0 forward and 1 is reverser
    # address 2 = 1 is OK and 0 is not okay
    # address 3 = 1 is PC available and 0 is not available
    # address 4 = 1 is PLC available and 0 is not available
    # address 6 = brush feed delay
    connect()
    if connection:
        # Write signal (0 or 1) to Coil (Discrete Output) at address 0
        # Replace with your actual address
        # Write value 1000 to Holding Register (D100)
        # register_address = 100  # Example: D100 in Delta PLC
        # value = 1000
        client.write_register(register_address, value)  # 'unit' is the slave ID

        # Read coil status
        response = client.read_holding_registers(register_address)

        if response.isError():
            print("Error reading Register")
        else:
            print(f"PLC Register Status: {response.registers[0]}")  
            # print(f"PLC Coil response: {response}")  

        print(f"Signal {value} sent to address {register_address}")
    else:
        print("Failed to connect to PLC")
    close()


def close():
    global client
    client.close()
    print("Disconnected to PLC")

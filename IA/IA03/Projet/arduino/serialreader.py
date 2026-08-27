import serial
import pandas as pd
import struct

# Configure serial port and other parameters
port = "COM5"          # Replace with the correct serial port
baud_rate = 9600       # Match the baud rate with the Arduino setup
timeout = 1            # Timeout for serial read

# Define arrays to store the x, y, and z data
SAMPLE_COUNT = 50
SAMPLE_LENGTH = 100

ser = serial.Serial(port, baud_rate, timeout=timeout)

# Write "s" to Serial

ser.write(b"s")

ser.set_buffer_size(rx_size = 40000000, tx_size = 12800)

with ser:
    x_array = []
    y_array = []
    z_array = []
    ids = []
    id_to_use = 0
    for v in range(SAMPLE_COUNT):
        for i in range(SAMPLE_LENGTH):
            data = ""
            while len(data) == 0 or data[-1] != "\n":
                data += ser.read(1).decode("utf-8")
                print(repr(data))
            x, y, z = [float(d) for d in str(data[:-1]).split("\t")]
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)
            ids.append(id_to_use)
        pd.DataFrame({"x": x_array, "y": y_array, "z": z_array, "id": ids}).to_csv("csv.csv", index=False)
        print(x, y, z)
        id_to_use += 1
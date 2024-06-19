        import smbus2
import time

# AMG8833 I2C address
AMG8833_I2C_ADDR = 0x69

# Registers
AMG8833_T01L = 0x80  # Temperature registers start at 0x80
AMG8833_PIXEL_ARRAY_SIZE = 64  # 8x8 pixel array

# Initialize I2C bus
bus = smbus2.SMBus(1)

def read_temperature():
    temp_data = []
    for i in range(AMG8833_PIXEL_ARRAY_SIZE):
        low_byte = bus.read_byte_data(AMG8833_I2C_ADDR, AMG8833_T01L + i * 2)
        high_byte = bus.read_byte_data(AMG8833_I2C_ADDR, AMG8833_T01L + i * 2 + 1)
        # Combine high and low bytes
        temperature = (high_byte << 8) | low_byte
        # Convert to signed 12-bit integer
        if temperature & 0x800:
            temperature -= 4096
        # Convert to Celsius
        temperature *= 0.25
        temp_data.append(temperature)
    return temp_data

def calculate_3x3_averages(temperatures):
    if len(temperatures) != AMG8833_PIXEL_ARRAY_SIZE:
        raise ValueError("Temperature data does not contain 64 elements.")
    
    averages = []
    for i in range(0, 8, 3):  # Rows: 0, 3, 6
        row = []
        for j in range(0, 8, 3):  # Columns: 0, 3, 6
            region = []
            for di in range(3):
                for dj in range(3):
                    if i + di < 8 and j + dj < 8:
                        region.append(temperatures[(i + di) * 8 + (j + dj)])
            avg_temp = sum(region) / len(region)
            row.append(avg_temp)
        averages.append(row)
    
    return averages

def print_3x3_averages():
    temperatures = read_temperature()
    averages = calculate_3x3_averages(temperatures)
    for row in averages:
        print(" ".join(f"[{avg:.2f}°C]" for avg in row))

try:
    while True:
        print_3x3_averages()
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    bus.close()

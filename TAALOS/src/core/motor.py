import serial
import time



class Motor:

    def __init__(self, PORT: str):
        self.BAUDRATE = 115200
        self.SERIAL_PORT = PORT
        self.ser = None


    def connect(self)->bool:
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE)
            time.sleep(2)  # Wait for the ESP32 to restart
            if self.ser.is_open:
                print(f"Connected to serial port {self.SERIAL_PORT} at {self.BAUDRATE} baud.")
                return True
            else:
                print(f"Error: Could not open serial port {self.SERIAL_PORT}.")
                return False
        except Exception as e:
            print(f"Error: Could not open serial port {self.SERIAL_PORT}: {e}")
            return False



    # def send_angle(self, angle: float):
    #     if self.ser and self.ser.is_open:
    #         command = f"{angle}\n"  # Encode the angle as bytes
    #         self.ser.write(f"{angle}\n".encode('utf-8'))
    #     else:
    #         print("Error: Serial port is not open. Call connect() first.")

    def send_angle(self, angle: float):
        if self.ser and self.ser.is_open:
            angle_int = int(round(angle))
            command = f"A{angle_int}\n"
            self.ser.write(command.encode('utf-8'))
        else:
            print("Error: Serial port is not open. Call connect() first.")

    def send_speed(self, rpm: float):
        """
        Send speed command to ESP32.

        Args:
            rpm: Speed in RPM (revolutions per minute)
        """
        if self.ser and self.ser.is_open:
            rpm_int = int(round(rpm))
            command = f"S{rpm_int}\n"
            self.ser.write(command.encode('utf-8'))
        else:
            print("Error: Serial port is not open. Call connect() first.")


if __name__ == "__main__":
    motor = Motor("/dev/tty.usbserial-0001")  # Replace with your serial port
    if motor.connect():
        motor.send_speed(10)  # Set speed to 20 RPM
        time.sleep(0.1)
        motor.send_angle(50)  # Send -90 degrees
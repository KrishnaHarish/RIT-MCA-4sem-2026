"""
data_generator.py — Sensor GPT | ELCIA COE
Generates a synthetic sensor dataset CSV mimicking real datasheet data.
"""

import pandas as pd
import random

random.seed(42)

SENSORS = [
    # Temperature & Humidity
    {"name": "DHT22", "type": "Temperature/Humidity", "protocol": "1-Wire", "voltage": "3.3-5V",
     "temp_range": "-40 to 80°C", "accuracy": "±0.5°C / ±2%RH", "i2c": False, "spi": False,
     "cost_usd": 3.5, "environment": "Indoor/Outdoor", "package": "Through-hole"},
    {"name": "DHT11", "type": "Temperature/Humidity", "protocol": "1-Wire", "voltage": "3.3-5V",
     "temp_range": "0 to 50°C", "accuracy": "±2°C / ±5%RH", "i2c": False, "spi": False,
     "cost_usd": 1.5, "environment": "Indoor", "package": "Through-hole"},
    {"name": "SHT31", "type": "Temperature/Humidity", "protocol": "I2C", "voltage": "2.4-5.5V",
     "temp_range": "-40 to 125°C", "accuracy": "±0.3°C / ±2%RH", "i2c": True, "spi": False,
     "cost_usd": 4.5, "environment": "Industrial", "package": "SMD"},
    {"name": "SHT40", "type": "Temperature/Humidity", "protocol": "I2C", "voltage": "1.8-3.6V",
     "temp_range": "-40 to 125°C", "accuracy": "±0.2°C / ±1.8%RH", "i2c": True, "spi": False,
     "cost_usd": 5.0, "environment": "Indoor/Industrial", "package": "SMD"},
    {"name": "AHT21", "type": "Temperature/Humidity", "protocol": "I2C", "voltage": "2.0-5.5V",
     "temp_range": "-40 to 85°C", "accuracy": "±0.3°C / ±2%RH", "i2c": True, "spi": False,
     "cost_usd": 2.0, "environment": "Indoor", "package": "SMD"},

    # Pressure & Altitude
    {"name": "BME280", "type": "Temperature/Pressure/Humidity", "protocol": "I2C/SPI", "voltage": "1.8-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±1 hPa / ±3%RH", "i2c": True, "spi": True,
     "cost_usd": 4.0, "environment": "Indoor/Weather", "package": "SMD"},
    {"name": "BMP388", "type": "Pressure/Altitude", "protocol": "I2C/SPI", "voltage": "1.65-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±0.2 hPa", "i2c": True, "spi": True,
     "cost_usd": 5.5, "environment": "Drone/UAV", "package": "SMD"},
    {"name": "BMP180", "type": "Pressure/Altitude", "protocol": "I2C", "voltage": "1.8-3.6V",
     "temp_range": "0 to 65°C", "accuracy": "±1 hPa", "i2c": True, "spi": False,
     "cost_usd": 2.0, "environment": "Indoor/Weather", "package": "SMD"},
    {"name": "MS5611", "type": "Pressure/Altitude", "protocol": "I2C/SPI", "voltage": "1.8-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±1.5 mbar", "i2c": True, "spi": True,
     "cost_usd": 8.0, "environment": "Aviation/Drone", "package": "SMD"},

    # Motion & IMU
    {"name": "MPU6050", "type": "Accelerometer/Gyroscope", "protocol": "I2C", "voltage": "2.3-3.4V",
     "temp_range": "-40 to 85°C", "accuracy": "±0.1°/s gyro", "i2c": True, "spi": False,
     "cost_usd": 3.0, "environment": "Robotics/Wearable", "package": "SMD"},
    {"name": "MPU9250", "type": "9-DOF IMU", "protocol": "I2C/SPI", "voltage": "2.4-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±0.01 dps gyro", "i2c": True, "spi": True,
     "cost_usd": 7.0, "environment": "Robotics/Drone", "package": "SMD"},
    {"name": "BMI088", "type": "Accelerometer/Gyroscope", "protocol": "I2C/SPI", "voltage": "1.7-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±0.004°/s gyro", "i2c": True, "spi": True,
     "cost_usd": 6.5, "environment": "Industrial/Drone", "package": "SMD"},
    {"name": "LSM6DS3", "type": "Accelerometer/Gyroscope", "protocol": "I2C/SPI", "voltage": "1.7-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±70 mdps", "i2c": True, "spi": True,
     "cost_usd": 5.0, "environment": "Wearable/IoT", "package": "SMD"},

    # Distance & Proximity
    {"name": "HC-SR04", "type": "Ultrasonic Distance", "protocol": "GPIO Trigger/Echo", "voltage": "5V",
     "temp_range": "-25 to 70°C", "accuracy": "±3mm", "i2c": False, "spi": False,
     "cost_usd": 1.0, "environment": "Robotics/Obstacle Detection", "package": "Through-hole"},
    {"name": "VL53L0X", "type": "ToF Distance", "protocol": "I2C", "voltage": "2.6-3.5V",
     "temp_range": "-20 to 70°C", "accuracy": "±3%", "i2c": True, "spi": False,
     "cost_usd": 6.0, "environment": "Robotics/Gesture", "package": "SMD"},
    {"name": "VL53L1X", "type": "ToF Distance", "protocol": "I2C", "voltage": "2.6-3.5V",
     "temp_range": "-20 to 70°C", "accuracy": "±1mm", "i2c": True, "spi": False,
     "cost_usd": 9.0, "environment": "Industrial/Automation", "package": "SMD"},
    {"name": "SHARP GP2Y0A21", "type": "IR Distance", "protocol": "Analog", "voltage": "5V",
     "temp_range": "-10 to 60°C", "accuracy": "±10%", "i2c": False, "spi": False,
     "cost_usd": 3.0, "environment": "Robotics", "package": "Through-hole"},

    # Gas & Air Quality
    {"name": "MQ-135", "type": "Air Quality (CO2/NH3)", "protocol": "Analog", "voltage": "5V",
     "temp_range": "-20 to 50°C", "accuracy": "±5%", "i2c": False, "spi": False,
     "cost_usd": 2.0, "environment": "Indoor Air Quality", "package": "Through-hole"},
    {"name": "MQ-7", "type": "Carbon Monoxide", "protocol": "Analog", "voltage": "5V",
     "temp_range": "-10 to 50°C", "accuracy": "±5%", "i2c": False, "spi": False,
     "cost_usd": 2.5, "environment": "Safety/Industrial", "package": "Through-hole"},
    {"name": "CCS811", "type": "eCO2/TVOC", "protocol": "I2C", "voltage": "1.8-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±15ppm CO2", "i2c": True, "spi": False,
     "cost_usd": 8.0, "environment": "Indoor Air Quality", "package": "SMD"},
    {"name": "SGP30", "type": "eCO2/TVOC", "protocol": "I2C", "voltage": "1.62-1.98V",
     "temp_range": "-40 to 85°C", "accuracy": "±15% CO2", "i2c": True, "spi": False,
     "cost_usd": 10.0, "environment": "Smart Home/IAQ", "package": "SMD"},
    {"name": "BME688", "type": "Gas/Temp/Humidity/Pressure", "protocol": "I2C/SPI", "voltage": "1.7-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±1 hPa / ±3%RH", "i2c": True, "spi": True,
     "cost_usd": 12.0, "environment": "Industrial/Smart Home", "package": "SMD"},

    # Light & UV
    {"name": "BH1750", "type": "Ambient Light", "protocol": "I2C", "voltage": "2.4-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±20%", "i2c": True, "spi": False,
     "cost_usd": 1.5, "environment": "Indoor/Outdoor", "package": "SMD"},
    {"name": "VEML7700", "type": "Ambient Light", "protocol": "I2C", "voltage": "2.5-3.6V",
     "temp_range": "-40 to 85°C", "accuracy": "±10%", "i2c": True, "spi": False,
     "cost_usd": 2.5, "environment": "IoT/Wearable", "package": "SMD"},
    {"name": "GUVA-S12SD", "type": "UV Index", "protocol": "Analog", "voltage": "2.7-5.5V",
     "temp_range": "-20 to 85°C", "accuracy": "±1 UV Index", "i2c": False, "spi": False,
     "cost_usd": 2.0, "environment": "Outdoor/Wearable", "package": "SMD"},

    # Soil & Water
    {"name": "Capacitive Soil Sensor v1.2", "type": "Soil Moisture", "protocol": "Analog", "voltage": "3.3-5V",
     "temp_range": "0 to 50°C", "accuracy": "±3%", "i2c": False, "spi": False,
     "cost_usd": 2.5, "environment": "Agriculture/Garden", "package": "Through-hole"},
    {"name": "DS18B20", "type": "Temperature (Waterproof)", "protocol": "1-Wire", "voltage": "3.0-5.5V",
     "temp_range": "-55 to 125°C", "accuracy": "±0.5°C", "i2c": False, "spi": False,
     "cost_usd": 2.5, "environment": "Industrial/Liquid", "package": "Probe"},
    {"name": "Atlas EZO pH", "type": "pH", "protocol": "I2C/UART", "voltage": "3.3-5V",
     "temp_range": "1 to 99°C", "accuracy": "±0.002 pH", "i2c": True, "spi": False,
     "cost_usd": 40.0, "environment": "Water Quality/Lab", "package": "Module"},

    # Current & Power
    {"name": "INA219", "type": "Current/Power Monitor", "protocol": "I2C", "voltage": "3.0-5.5V",
     "temp_range": "-40 to 125°C", "accuracy": "±1% current", "i2c": True, "spi": False,
     "cost_usd": 2.0, "environment": "Power Management", "package": "SMD"},
    {"name": "INA3221", "type": "3-Channel Current Monitor", "protocol": "I2C", "voltage": "2.7-5.5V",
     "temp_range": "-40 to 125°C", "accuracy": "±0.8% current", "i2c": True, "spi": False,
     "cost_usd": 4.5, "environment": "Multichannel Power", "package": "SMD"},

    # Sound
    {"name": "INMP441", "type": "MEMS Microphone", "protocol": "I2S", "voltage": "1.8-3.3V",
     "temp_range": "-40 to 85°C", "accuracy": "±1 dB SNR", "i2c": False, "spi": False,
     "cost_usd": 3.5, "environment": "Audio/Voice", "package": "SMD"},
    {"name": "MAX4466", "type": "Sound/Microphone", "protocol": "Analog", "voltage": "2.4-5.5V",
     "temp_range": "-40 to 85°C", "accuracy": "±3 dB", "i2c": False, "spi": False,
     "cost_usd": 2.0, "environment": "Audio Detection", "package": "Module"},
]


def generate_dataset():
    records = []
    for s in SENSORS:
        protocols = []
        if s["i2c"]:
            protocols.append("I2C")
        if s["spi"]:
            protocols.append("SPI")
        if not protocols:
            protocols.append(s["protocol"])

        records.append({
            "Sensor Name": s["name"],
            "Sensor Type": s["type"],
            "Communication Protocol": s["protocol"],
            "I2C Compatible": "Yes" if s["i2c"] else "No",
            "SPI Compatible": "Yes" if s["spi"] else "No",
            "Operating Voltage": s["voltage"],
            "Temperature Range": s["temp_range"],
            "Accuracy": s["accuracy"],
            "Cost (USD)": s["cost_usd"],
            "Target Environment": s["environment"],
            "Package Type": s["package"],
            "Description": (
                f"The {s['name']} is a {s['type']} sensor that communicates via {s['protocol']}. "
                f"It supports {s['voltage']} supply voltage and operates across {s['temp_range']}. "
                f"Accuracy is {s['accuracy']}. It is well-suited for {s['environment']} applications. "
                f"Cost: approximately ${s['cost_usd']} USD. Package: {s['package']}. "
                f"I2C: {'Yes' if s['i2c'] else 'No'}. SPI: {'Yes' if s['spi'] else 'No'}."
            )
        })

    df = pd.DataFrame(records)
    df.to_csv("sensors_dataset.csv", index=False)
    print(f"✅ Generated sensors_dataset.csv with {len(df)} sensors.")
    return df


if __name__ == "__main__":
    generate_dataset()

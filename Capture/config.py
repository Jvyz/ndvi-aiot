#!/usr/bin/python
# -*- coding:utf-8 -*-
import serial
try:
    # Try the Pi 5 compatible GPIO library first
    import RPi.GPIO as GPIO
    print("Using standard RPi.GPIO library")
except RuntimeError as e:
    if "SOC peripheral base address" in str(e):
        # Fallback for Pi 5 - use lgpio directly
        print("Pi 5 detected, using lgpio backend")
        import RPi.GPIO as GPIO
        # Force use of lgpio backend
        import os
        os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

Temp = '0123456789ABCDEF*'

class config(object):
    FORCE  = 17
    STANDBY= 4
    
    def __init__(self, Baudrate = 9600):
        # Initialize serial first - Pi 5 uses ttyAMA0
        self.serial = serial.Serial("/dev/ttyAMA0", Baudrate, timeout=1)
        
        # GPIO setup with error handling for Pi 5
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup pins with proper initial states
            GPIO.setup(self.STANDBY, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(self.FORCE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print("GPIO initialized successfully")
            
        except Exception as e:
            print(f"GPIO initialization error: {e}")
            print("Continuing without GPIO control...")
            self.gpio_available = False
        else:
            self.gpio_available = True
        
    def Uart_SendByte(self, value): 
        self.serial.write(value) 
        
    def Uart_SendString(self, value): 
        self.serial.write(value)
  
    def Uart_ReceiveByte(self): 
        return self.serial.read(1)

    def Uart_ReceiveString(self, value): 
        data = self.serial.read(value)
        return data
        
    def Uart_Set_Baudrate(self, Baudrate):
        self.serial.close()
        self.serial = serial.Serial("/dev/ttyAMA0", Baudrate, timeout=1)
        
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'serial') and self.serial.is_open:
            self.serial.close()
        if self.gpio_available:
            try:
                GPIO.cleanup()
            except:
                pass

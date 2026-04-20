try:
    import RPi.GPIO as GPIO
except RuntimeError as e:
    if "SOC peripheral base address" in str(e):
        print("Pi 5 detected, using lgpio backend")
        import RPi.GPIO as GPIO
        import os
        os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import config
import math
import time
from micropyGPS import MicropyGPS

g=MicropyGPS(+8)
Temp = '0123456789ABCDEF*'
BUFFSIZE = 1100

pi = 3.14159265358979324
a = 6378245.0
ee = 0.00669342162296594323
x_pi = 3.14159265358979324 * 3000.0 / 180.0

class L76X(object):
    Lon = 0.0
    Lat = 0.0
    Lon_area = 'E'
    Lat_area = 'W'
    Time_H = 0
    Time_M = 0
    Time_S = 0
    Status = 0
    Lon_Baidu = 0.0
    Lat_Baidu = 0.0
    Lon_Google = 0.0
    Lat_Google = 0.0
    
    GPS_Lon = 0
    GPS_Lat = 0
    
    # GPS quality indicators
    satellites_in_use = 0
    hdop = 99.9
    fix_quality = 0
    
    #Startup mode
    SET_HOT_START       = '$PMTK101'
    SET_WARM_START      = '$PMTK102'
    SET_COLD_START      = '$PMTK103'
    SET_FULL_COLD_START = '$PMTK104'

    #Standby mode -- Exit requires high level trigger
    SET_PERPETUAL_STANDBY_MODE      = '$PMTK161'

    SET_PERIODIC_MODE               = '$PMTK225'
    SET_NORMAL_MODE                 = '$PMTK225,0'
    SET_PERIODIC_BACKUP_MODE        = '$PMTK225,1,1000,2000'
    SET_PERIODIC_STANDBY_MODE       = '$PMTK225,2,1000,2000'
    SET_PERPETUAL_BACKUP_MODE       = '$PMTK225,4'
    SET_ALWAYSLOCATE_STANDBY_MODE   = '$PMTK225,8'
    SET_ALWAYSLOCATE_BACKUP_MODE    = '$PMTK225,9'

    #Set the message interval,100ms~10000ms
    SET_POS_FIX         = '$PMTK220'
    SET_POS_FIX_100MS   = '$PMTK220,100'
    SET_POS_FIX_200MS   = '$PMTK220,200'
    SET_POS_FIX_400MS   = '$PMTK220,400'
    SET_POS_FIX_800MS   = '$PMTK220,800'
    SET_POS_FIX_1S      = '$PMTK220,1000'
    SET_POS_FIX_2S      = '$PMTK220,2000'
    SET_POS_FIX_4S      = '$PMTK220,4000'
    SET_POS_FIX_8S      = '$PMTK220,8000'
    SET_POS_FIX_10S     = '$PMTK220,10000'

    #Switching time output
    SET_SYNC_PPS_NMEA_OFF   = '$PMTK255,0'
    SET_SYNC_PPS_NMEA_ON    = '$PMTK255,1'

    #To restore the system default setting
    SET_REDUCTION               = '$PMTK314,-1'

    #Set NMEA sentence output frequencies 
    SET_NMEA_OUTPUT = '$PMTK314,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0'
    #Baud rate
    SET_NMEA_BAUDRATE          = '$PMTK251'
    SET_NMEA_BAUDRATE_115200   = '$PMTK251,115200'
    SET_NMEA_BAUDRATE_57600    = '$PMTK251,57600'
    SET_NMEA_BAUDRATE_38400    = '$PMTK251,38400'
    SET_NMEA_BAUDRATE_19200    = '$PMTK251,19200'
    SET_NMEA_BAUDRATE_14400    = '$PMTK251,14400'
    SET_NMEA_BAUDRATE_9600     = '$PMTK251,9600'
    SET_NMEA_BAUDRATE_4800     = '$PMTK251,4800'
    
    # Timeout constant for the L76X_Gat_GNRMC loop (in seconds of attempted reads)
    GPS_READ_TIMEOUT = 10 

    def __init__(self):
        # 1. Initialize at default 9600 (or factory default)
        self.config = config.config(9600)
        
        # 2. Reconfigure module to 115200 baud (matching main.py's successful configuration)
        print("Reconfiguring GPS module to 115200 baud...")
        self.L76X_Send_Command(self.SET_NMEA_BAUDRATE_115200)
        time.sleep(1) 
        
        # 3. Update the UART connection to the new 115200 baudrate
        self.config.Uart_Set_Baudrate(115200)
        print("UART baudrate set to 115200.")
        
        # 4. Set position fix rate (using 1S for stability, as 400MS might stress the tiny thread)
        self.L76X_Send_Command(self.SET_POS_FIX_1S)
        time.sleep(0.5)
        
        # 5. Set NMEA output sentences
        self.L76X_Send_Command(self.SET_NMEA_OUTPUT)
        time.sleep(0.5)
        
        # Ensure initial state is clean
        self.L76X_Exit_BackupMode()

    def L76X_Send_Command(self, data):
        Check = ord(data[1]) 
        for i in range(2, len(data)):
            Check = Check ^ ord(data[i]) 
        data = data + Temp[16]
        data = data + Temp[(Check//16)]
        data = data + Temp[(Check%16)]
        self.config.Uart_SendString(data.encode())
        self.config.Uart_SendByte('\r'.encode())
        self.config.Uart_SendByte('\n'.encode())
        print(data)
    
    def parse_gga_sentence(self, gga_line):
        """Parse NMEA GGA sentence for accurate coordinates and quality data"""
        try:
            parts = gga_line.split(',')
            
            if len(parts) >= 15 and (parts[0] == '$GNGGA' or parts[0] == '$GPGGA'):
                # Extract quality indicators
                self.fix_quality = int(parts[6]) if parts[6] else 0
                self.satellites_in_use = int(parts[7]) if parts[7] else 0
                self.hdop = float(parts[8]) if parts[8] else 99.9
                
                # Extract coordinates if we have a fix
                if self.fix_quality > 0 and parts[2] and parts[4]:
                    # Parse latitude (DDMM.MMMMMM format)
                    lat_nmea = parts[2]
                    lat_dir = parts[3]
                    lat_degrees = int(lat_nmea[:2])
                    lat_minutes = float(lat_nmea[2:])
                    self.Lat = lat_degrees + (lat_minutes / 60.0)
                    if lat_dir == 'S':
                        self.Lat = -self.Lat
                    
                    # Parse longitude (DDDMM.MMMMMM format)
                    lon_nmea = parts[4] 
                    lon_dir = parts[5]
                    lon_degrees = int(lon_nmea[:3])
                    lon_minutes = float(lon_nmea[3:])
                    self.Lon = lon_degrees + (lon_minutes / 60.0)
                    if lon_dir == 'W':
                        self.Lon = -self.Lon
                    
                    return True
                    
        except Exception as e:
            print(f"Error parsing GGA sentence: {e}")
        
        return False
        
    def L76X_Gat_GNRMC(self):
        data=''
        gga_found = False
        
        # --- FIX: Use a timeout mechanism to prevent thread hanging ---
        attempts = 0
        
        # Loop for a fixed number of attempts (10 attempts * 1s timeout = 10s max wait)
        while attempts < self.GPS_READ_TIMEOUT: 
            if g.valid:
                self.Status = 1
            else:
                self.Status = 0
            
            x=self.config.Uart_ReceiveByte() 
            
            if x == b'':
                # UART timed out (config.py sets timeout=1s)
                attempts += 1
                time.sleep(0.1) # Wait a short moment before next check
                continue 
            
            # --- FIX: Handle non-ASCII bytes and decoding errors ---
            if x==b'$':
                line = x.decode(errors='ignore') # Start with '$'
                data += line

                # Reading the rest of the line (still relying on blocking read here)
                while True:
                    x=self.config.Uart_ReceiveByte()
                    
                    if x == b'':
                        # Timeout while reading the line - break and process next attempt
                        break
                        
                    if x==b'\r':
                        data+='\r\n'
                        break
                    
                    try:
                        decoded_char = x.decode() 
                        line += decoded_char
                        data += decoded_char
                        g.update(decoded_char)
                    except UnicodeDecodeError as e:
                        # Skip bad bytes
                        continue
                        
                
                # Reset attempts on successfully started sentence
                attempts = 0 
                
                # Parse GGA sentence for accurate coordinates
                if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                    if self.parse_gga_sentence(line):
                        gga_found = True
                        # Update time from micropyGPS
                        self.Time_H = g.timestamp[0]
                        self.Time_M = g.timestamp[1] 
                        self.Time_S = g.timestamp[2]
                        # Set status based on fix quality
                        self.Status = 1 if self.fix_quality > 0 and self.satellites_in_use >= 4 else 0
                
                # Break on GNGLL to complete the cycle and return control 
                if '$GNGLL' in line:
                    break
            
            attempts += 1
        
        # --- End of FIX ---

        # Fallback to micropyGPS if GGA parsing failed
        if not gga_found and g.valid:
            # Correct coordinate conversion from micropyGPS
            self.Lat = g.latitude[0] + (g.latitude[1] / 60.0) 
            self.Lon = g.longitude[0] + (g.longitude[1] / 60.0)
            
            if g.latitude[2] != 'N':
                self.Lat = self.Lat * (-1)
            if g.longitude[2] != 'E':
                self.Lon = self.Lon * (-1)
            
            self.Time_H = g.timestamp[0]
            self.Time_M = g.timestamp[1]
            self.Time_S = g.timestamp[2]
            
            if self.Lat != 0.0 and self.Lon != 0.0:
                 self.Status = 1 # Mark status as 1 if micropyGPS has valid data

        # Print data received during this 10-second window (may be empty if timed out)
        # print(data) 
        pass # Suppress excessive printing inside the tight thread loop
        data='\r\n'

    def transformLat(self, x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 *math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret

    def transformLon(self, x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret

    def bd_encrypt(self):
        x = self.Lon_Google  
        y = self.Lat_Google  
        z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * x_pi)
        theta = math.atan2(y, x) + 0.000003 * math.cos(x * x_pi)
        self.Lon_Baidu = z * math.cos(theta) + 0.0065
        self.Lat_Baidu = z * math.sin(theta) + 0.006

    def transform(self):
        dLat = self.transformLat(self.GPS_Lon - 105.0, self.GPS_Lat - 35.0)
        dLon = self.transformLon(self.GPS_Lon - 105.0, self.GPS_Lat - 35.0)
        radLat = self.GPS_Lat / 180.0 * pi
        magic = math.sin(radLat)
        magic = 1 - ee * magic * magic
        sqrtMagic = math.sqrt(magic)  
        dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
        dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * pi)
        self.Lat_Google = self.GPS_Lat + dLat  
        self.Lon_Google = self.GPS_Lon + dLon 

    def L76X_Baidu_Coordinates(self, U_Lat, U_Lon):
        self.GPS_Lat = U_Lat
        self.GPS_Lon = U_Lon
        self.transform()
        self.bd_encrypt()

    def L76X_Google_Coordinates(self, U_Lat, U_Lon):
        self.GPS_Lat = U_Lat
        self.GPS_Lon = U_Lon
        self.transform()

    def L76X_Set_Baudrate(self, Baudrate):
        self.config.Uart_Set_Baudrate(Baudrate)

    def L76X_Exit_BackupMode(self):
        if hasattr(self.config, 'gpio_available') and self.config.gpio_available:
            try:
                GPIO.setup(self.config.FORCE, GPIO.OUT)
                time.sleep(1)
                GPIO.output(self.config.FORCE, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(self.config.FORCE, GPIO.LOW)
                time.sleep(1)
                GPIO.setup(self.config.FORCE, GPIO.IN)
                print("GPS backup mode exit completed using GPIO")
            except Exception as e:
                print(f"GPIO backup mode exit failed: {e}")
                print("Continuing without GPIO backup mode exit...")
        else:
            print("GPIO not available, skipping backup mode exit")
    
    def get_gps_quality_info(self):
        """Return GPS quality information for debugging"""
        return {
            'fix_quality': self.fix_quality,
            'satellites': self.satellites_in_use,
            'hdop': self.hdop,
            'status': self.Status,
            'latitude': self.Lat,
            'longitude': self.Lon
        }
    
    def validate_coordinates(self):
        """Validate if coordinates are reasonable for Vietnam/HCMC area"""
        # Vietnam coordinate bounds (approximate)
        lat_valid = 8.0 <= self.Lat <= 24.0
        lon_valid = 102.0 <= self.Lon <= 110.0
        
        return lat_valid and lon_valid

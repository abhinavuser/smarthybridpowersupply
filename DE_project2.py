import time
import csv

with open(r"C:\Users\chipn\OneDrive\Desktop\excel.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:

            Solar_Panel_Output = float(row['Solar_Panel_Output'])  
            Battery_Charge = float(row['Battery_Charge']) 
            Battery_Discharge = float(row['Battery_Discharge']) 
            Household_Load = float(row['Household_Load'])  
        except KeyError as e:
            pass
            
       
def generate_solar_power():
  
    return 200


def manage_battery_power(Battery_Charge, Battery_Discharge):
    a=[]
    for i in Battery_Charge:
        for j in Battery_Discharge:
            a.append(i-j)
    return  a

def household_load():

    return Household_Load  


while True:
    solar_power = generate_solar_power()
    battery_power = manage_battery_power(Battery_Charge, Battery_Discharge)  
    total_power = solar_power + battery_power

    ac_power = total_power * 0.9

    if ac_power >= household_load():
        print(f"Supplying {household_load()} Watts to the household.")
    else:
        print(f"Insufficient power. Running on grid electricity.")

    time.sleep(0.5)

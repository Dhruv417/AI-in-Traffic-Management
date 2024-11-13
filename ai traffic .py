#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import threading

# Load and preprocess the dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Direction'] = df['Zone'].apply(lambda x: "North" if x <= 36 else 
                                                 "East" if x <= 72 else 
                                                 "South" if x <= 108 else 
                                                 "West")
    zones = df.pivot_table(values='Traffic', index=['Date', 'Weather'], columns='Direction', fill_value=0)
    zones.columns = ["North_Flow", "East_Flow", "South_Flow", "West_Flow"]
    zones.reset_index(inplace=True)
    zones['Optimal_Green'] = zones[["North_Flow", "South_Flow", "East_Flow", "West_Flow"]].idxmax(axis=1)
    zones['Optimal_Green'] = zones['Optimal_Green'].map({"North_Flow": 0, "South_Flow": 1, "East_Flow": 2, "West_Flow": 3})
    return zones

def train_model(data):
    X = data.drop(columns=["Date", "Optimal_Green"])
    y = data["Optimal_Green"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model, X.columns

class TrafficLight:
    def __init__(self, location):
        self.location = location
        self.status = {"North": "red", "South": "red", "East": "red", "West": "red"}
        self.green_time = 30
        self.timer_thread = None
        self.total_cars_cleared = {"North": 0, "South": 0, "East": 0, "West": 0}
        self.total_waiting_time = {"North": 0, "South": 0, "East": 0, "West": 0}

    def change_status(self, direction, new_status, traffic_flow, duration=None):
        if self.status[direction] == new_status:
            return
        self.status[direction] = new_status
        if new_status == "green":
            cars_cleared = traffic_flow[direction]
            self.total_cars_cleared[direction] += cars_cleared
            wait_time = self.calculate_waiting_time(traffic_flow)
            self.total_waiting_time[direction] += wait_time
            if duration:
                self.green_time = duration
            if self.timer_thread and self.timer_thread.is_alive():
                self.timer_thread.cancel()
            self.timer_thread = threading.Timer(self.green_time, self.reset_lights)
            self.timer_thread.start()

    def reset_lights(self):
        for direction in self.status.keys():
            self.status[direction] = "red"

    def calculate_waiting_time(self, traffic_flow):
        return sum(traffic_flow.values()) / 10

    def display_metrics(self):
        for direction in self.total_cars_cleared:
            avg_waiting = self.total_waiting_time[direction] / (self.total_cars_cleared[direction] or 1)
            print(f"Direction {direction}: Total Cars Cleared = {self.total_cars_cleared[direction]}, "
                  f"Average Waiting Time = {avg_waiting:.2f} seconds")

class Sensor:
    def __init__(self, location):
        self.location = location

    def measure_traffic_flow(self):
        return {
            "North": random.randint(0, 100),
            "South": random.randint(0, 100),
            "East": random.randint(0, 100),
            "West": random.randint(0, 100)
        }

    def measure_weather_conditions(self):
        return random.choice(["sunny", "rainy", "snowy"])

class SmartTrafficSystem:
    def __init__(self, model, feature_columns):
        self.intersections = {}
        self.model = model
        self.feature_columns = feature_columns
        self.metric_display_interval = 30
        self.last_metric_display = time.time()
        self.force_green_counter = 0  # Counter to force periodic green for East and West

    def add_intersection(self, location, area):
        self.intersections[location] = {
            "traffic_light": TrafficLight(location),
            "sensor": Sensor(location),
            "area": area
        }

    def run(self):
        while True:
            for intersection, details in self.intersections.items():
                traffic_flow = details["sensor"].measure_traffic_flow()
                weather_condition = details["sensor"].measure_weather_conditions()
                
                X_input = pd.DataFrame([{
                    "North_Flow": traffic_flow["North"],
                    "South_Flow": traffic_flow["South"],
                    "East_Flow": traffic_flow["East"],
                    "West_Flow": traffic_flow["West"],
                    "Weather": {"sunny": 0, "rainy": 1, "snowy": 2}[weather_condition]
                }], columns=self.feature_columns)
                
                optimal_direction_index = self.model.predict(X_input)[0]
                optimal_direction = ["North", "South", "East", "West"][optimal_direction_index]
                
                # Force East or West green every 5 cycles if not selected
                if self.force_green_counter >= 5 and optimal_direction in ["North", "South"]:
                    optimal_direction = random.choice(["East", "West"])
                    self.force_green_counter = 0
                else:
                    self.force_green_counter += 1
                
                dynamic_green_time = max(10, min(30, traffic_flow[optimal_direction] // 5))
                self.update_traffic_lights(details["traffic_light"], optimal_direction, traffic_flow, dynamic_green_time)
                
                # Print real-time intersection status
                print("\n" + "="*50)
                print(f" Intersection: {intersection} | Area: {details['area']}")
                print(f" Weather: {weather_condition.capitalize()} | Optimal Green Direction: {optimal_direction}")
                print("-" * 50)
                print(" Traffic Light Status:")
                for direction, status in details["traffic_light"].status.items():
                    light_color = status.upper()
                    print(f"    ▶ {direction} direction: {light_color}")

            # Display cumulative performance metrics periodically
            self.display_cumulative_metrics()
            time.sleep(5)

    def update_traffic_lights(self, traffic_light, optimal_direction, traffic_flow, duration):
        for direction in traffic_light.status.keys():
            if direction == optimal_direction:
                traffic_light.change_status(direction, "green", traffic_flow, duration)
            else:
                traffic_light.change_status(direction, "red", traffic_flow)

    def display_cumulative_metrics(self):
        # Check if 30 seconds have passed since the last display
        if time.time() - self.last_metric_display >= self.metric_display_interval:
            print("\n\n--- Cumulative Performance Metrics ---")
            for intersection, details in self.intersections.items():
                print(f"\nMetrics for Intersection: {intersection}")
                details["traffic_light"].display_metrics()
            print("====================================================\n\n")
            self.last_metric_display = time.time()  # Update the last display time

# Run the system
# Run the system
if __name__ == "__main__":
    file_path = 'Dataset.csv'
    data = preprocess_data(file_path)
    model, feature_columns = train_model(data)
    system = SmartTrafficSystem(model, feature_columns)
    
    # Adding intersections with diverse areas
    system.add_intersection("City Center", "Central Business District")
    system.add_intersection("Suburban Area", "Residential Zone")
    system.add_intersection("East Park", "Recreational Area")
    system.add_intersection("Industrial Zone", "Manufacturing District")
    system.add_intersection("University District", "Educational Zone")
    
    # Start the traffic management system
    system.run()


# In[ ]:





# In[ ]:





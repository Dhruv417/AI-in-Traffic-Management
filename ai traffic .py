import pandas as pd
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import threading


def preprocess_data_new(file_path):
    df = pd.read_csv(file_path)
    df['Optimal_Green'] = df[
        ['North_Flow', 'South_Flow', 'East_Flow', 'West_Flow']
    ].idxmax(axis=1)
    df['Optimal_Green'] = df['Optimal_Green'].map(
        {"North_Flow": 0, "South_Flow": 1, "East_Flow": 2, "West_Flow": 3}
    )
    scaler = StandardScaler()
    numeric_features = [
        "North_Flow", "South_Flow", "East_Flow", "West_Flow",
        "Queue_Length_North", "Queue_Length_South",
        "Queue_Length_East", "Queue_Length_West"
    ]
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df

# Train the model
def train_model_new(data):
    X = data.drop(columns=["Optimal_Green"])
    y = data["Optimal_Green"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n                       📈 Model Performance📈 ")
    print(classification_report(y_test, y_pred))
    #print("=========================\n")
    return model, X.columns
    print("\n")

# Traffic Light Class
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
            print(f"    ▶ Direction {direction}: Total Cars Cleared = {self.total_cars_cleared[direction]}, "
                  f"Average Waiting Time = {avg_waiting:.2f} seconds")

# Sensor Class
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

# Smart Traffic System Class with Emergency Handling
class SmartTrafficSystemWithEmergency:
    def __init__(self, model, feature_columns):
        self.intersections = {}
        self.model = model
        self.feature_columns = feature_columns
        self.metric_display_interval = 30
        self.last_metric_display = time.time()

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
                queue_lengths = {dir_: random.randint(0, 20) for dir_ in ["North", "South", "East", "West"]}

                # Reduced frequency of emergency vehicles (5% probability per direction)
                emergency_priorities = {
                    dir_: 1 if random.random() < 0.05 else 0  # 5% chance of emergency
                    for dir_ in ["North", "South", "East", "West"]
                }

                # Prioritize emergency vehicles if any
                emergency_direction = self.prioritize_emergency(emergency_priorities)
                if emergency_direction:
                    print(f"🚨 Emergency detected at {intersection} in {emergency_direction} direction!")
                    print(f"🛑 Notifying nearby intersections to clear the path.\n")
                    optimal_direction = emergency_direction
                    dynamic_green_time = 30  # Maximum green light duration for emergencies
                else:
                    # Prepare model input for non-emergency scenarios
                    X_input = pd.DataFrame([{
                        "North_Flow": traffic_flow["North"],
                        "South_Flow": traffic_flow["South"],
                        "East_Flow": traffic_flow["East"],
                        "West_Flow": traffic_flow["West"],
                        "Weather": {"sunny": 0, "rainy": 1, "snowy": 2}[weather_condition],
                        "Queue_Length_North": queue_lengths["North"],
                        "Queue_Length_South": queue_lengths["South"],
                        "Queue_Length_East": queue_lengths["East"],
                        "Queue_Length_West": queue_lengths["West"],
                        "Emergency_Priority_North": emergency_priorities["North"],
                        "Emergency_Priority_South": emergency_priorities["South"],
                        "Emergency_Priority_East": emergency_priorities["East"],
                        "Emergency_Priority_West": emergency_priorities["West"],
                    }], columns=self.feature_columns)

                    # Validate and clean the input
                    X_input = X_input.fillna(0)  # Replace NaN values with 0
                    X_input = X_input.replace([float('inf'), -float('inf')], 0)  # Replace infinities with 0

                    # Pass validated input to the model
                    try:
                        optimal_direction_index = self.model.predict(X_input)[0]
                        optimal_direction = ["North", "South", "East", "West"][optimal_direction_index]
                    except Exception as e:
                        print(f"❌ Error during prediction: {e}")
                        continue

                    # Adjust green time dynamically based on weather
                    weather_adjustment = self.adjust_for_weather(weather_condition)
                    dynamic_green_time = max(10, min(30, traffic_flow[optimal_direction] // 5)) + weather_adjustment

                self.update_traffic_lights(details["traffic_light"], optimal_direction, traffic_flow, dynamic_green_time)

                # Display Traffic Light Status
                
                print(f"🌐 Intersection: {intersection} | Area: {details['area']}")
                print(f"🌦️ Weather: {weather_condition.capitalize()} | 🟩 Optimal Green Direction: {optimal_direction}")
                print(f"⏱️ Green Light Duration: {dynamic_green_time} seconds")
                print("-" * 50)
                print("🚦 Traffic Light Status:")
                for direction, status in details["traffic_light"].status.items():
                    light_color = "🟢 GREEN" if status == "green" else "🔴 RED"
                    print(f"    ▶ {direction} Direction: {light_color}")
                print("\n" + "=" * 110)
                print("\n")
            self.display_cumulative_metrics()
            time.sleep(5)

    def prioritize_emergency(self, emergency_priorities):
        for direction, priority in emergency_priorities.items():
            if priority:  # Emergency detected
                return direction
        return None  # No emergency detected

    def adjust_for_weather(self, weather_condition):
        if weather_condition in ["rainy", "snowy"]:
            print(f"🌧️ Adjusting green light time for adverse weather: {weather_condition}")
            return 10  # Extend green light time by 10 seconds
        return 0  # No adjustment for clear weather

    def update_traffic_lights(self, traffic_light, optimal_direction, traffic_flow, duration):
        for direction in traffic_light.status.keys():
            if direction == optimal_direction:
                traffic_light.change_status(direction, "green", traffic_flow, duration)
            else:
                traffic_light.change_status(direction, "red", traffic_flow)

    def display_cumulative_metrics(self):
        if time.time() - self.last_metric_display >= self.metric_display_interval:
            print("\n=== 🚘 Cumulative Performance Metrics 🚘 ===")
            for intersection, details in self.intersections.items():
                print(f"\n📍 Metrics for Intersection: {intersection}")
                details["traffic_light"].display_metrics()
            print("\n" + "=" * 110)
            self.last_metric_display = time.time()


file_path = 'synthetic_traffic_datas.csv'
data_new = preprocess_data_new(file_path)
model_new, feature_columns_new = train_model_new(data_new)
system = SmartTrafficSystemWithEmergency(model_new, feature_columns_new)


system.add_intersection("City Center", "Central Business District")
system.add_intersection("Suburban Area", "Residential Zone")
system.add_intersection("East Park", "Recreational Area")
system.add_intersection("Industrial Zone", "Manufacturing District")
system.add_intersection("University District", "Educational Zone")


system.run()

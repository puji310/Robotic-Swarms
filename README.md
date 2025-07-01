# Robotic-Swarms
Robotic swarms for search operations
🐝 Swarm Robotics Simulation with Dynamic Obstacles and Target Discovery
This project simulates a team of autonomous robots navigating a 2D environment to discover and report targets while avoiding dynamic obstacles and maintaining formation. It features leadership switching, failure handling, obstacle avoidance, formation control, and target discovery visualization.
📌 Features
✅ Multi-robot coordination (formation keeping with a leader-follower model)
🎯 Target search and discovery
📡 Communication and information sharing
⚠️ Dynamic obstacle avoidance
🔁 Leadership scoring and switching
🧭 Exploration behavior
💥 Random failure simulation
📊 Visualization of robot trajectories and position vs. time plots
🛠 Environment Setup
-Requirements:
-Python 3.7+
-NumPy
-Matplotlib
🚀 Running the Simulation
Robots will explore the environment, detect and share target information.
After all targets are discovered, robots return to their initial starting position.
The simulation window shows:
-Robots (🔵 followers, 🔴 leader, ❌ failed)
-Moving obstacles (⬛)
-Discovered/undiscovered targets (🟢/🔴)
-Communication and sensing ranges
-Exploration paths and formation lines
📈 Output
At the end of the simulation, a graph robot_position_graphs.png is generated showing each robot’s X/Y position over time with:
✅ Start and End points
⭐ Target discovery timestamps
❌ Failure points (if any)
🧠 Key Concepts Demonstrated
-Swarm Intelligence
-Leader Election and Role Assignment
-Dynamic Environments
-Formation Control & Recovery
-Autonomous Navigation & Exploration
-Fault Tolerance in Multi-Agent Systems

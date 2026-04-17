from openai import OpenAI

# Step 1: Initialize the client to point to YOUR local computer
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Step 2: Use your structured data (this would eventually come from Step 3)
traffic_data = {
    "level": "High",
    "speed": 12,
    "vehicle": "Ambulance",
    "goal": "Fastest arrival"
}

# Step 3: Send the structured prompt to Llama 7B
completion = client.chat.completions.create(
  model="local-model", # LM Studio uses whatever model you loaded
  messages=[
    {"role": "system", "content": "You are a traffic routing expert. Give concise, simple advice."},
    {"role": "user", "content": f"Data: {traffic_data}. Suggest a route and explain why simply."}
  ]
)

# Step 4: Show the output (This is the Output of Step 4)
print("\n--- AI Route Recommendation ---")
print(completion.choices[0].message.content)
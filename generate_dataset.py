import csv
import random

# Set seed for reproducibility
random.seed(42)

prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega", "Quantum", "Nexus", "Aero", "Stellar", "Nova", "Cyber", "Synapse", "Apex", "Zenith", "Vanguard", "Pinnacle", "Aether", "Luminous", "Vortex", "Titan", "Echo"]
suffixes = ["Corp", "Industries", "Tech", "Startup", "Logistics", "Solutions", "Enterprises", "Systems", "Innovations", "Global", "Networks", "Ventures", "Dynamics", "Holdings", "Group", "Labs", "Studios", "Partners", "Capital", "Media"]

def generate_company_name():
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"

def generate_data(num_samples):
    data = []
    names = set()
    while len(data) < num_samples:
        name = generate_company_name()
        # Ensure uniqueness
        if name in names:
            name = f"{name} {random.randint(1, 1000)}"
        names.add(name)

        is_profitable = random.choices([0, 1], weights=[0.3, 0.7])[0]
        has_high_debt = random.choices([0, 1], weights=[0.6, 0.4])[0]
        has_strong_cash_flow = random.choices([0, 1], weights=[0.4, 0.6])[0]
        has_good_management = random.choices([0, 1], weights=[0.5, 0.5])[0]
        has_regulatory_issues = random.choices([0, 1], weights=[0.8, 0.2])[0]

        # Logic for recommend_buy
        recommend_buy = 1 if (is_profitable == 1 and
                              has_high_debt == 0 and
                              has_regulatory_issues == 0 and
                              (has_strong_cash_flow == 1 or has_good_management == 1)) else 0

        data.append({
            "company": name,
            "is_profitable": is_profitable,
            "has_high_debt": has_high_debt,
            "has_strong_cash_flow": has_strong_cash_flow,
            "has_good_management": has_good_management,
            "has_regulatory_issues": has_regulatory_issues,
            "recommend_buy": recommend_buy
        })

    return data

def save_to_csv(data, filename):
    keys = data[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

# Generate datasets
train_data = generate_data(300)
test_data = generate_data(100)

# Save to CSV
save_to_csv(train_data, "train.csv")
save_to_csv(test_data, "test.csv")

print("train.csv and test.csv generated successfully.")

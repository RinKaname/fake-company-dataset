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

        # GENERATE FUZZY CONTINUOUS VARIABLES (0.0 to 1.0)
        # Instead of strict 0 or 1, we use a continuous scale.
        is_profitable = round(random.uniform(0.0, 1.0), 3)
        has_high_debt = round(random.uniform(0.0, 1.0), 3)
        has_strong_cash_flow = round(random.uniform(0.0, 1.0), 3)
        has_good_management = round(random.uniform(0.0, 1.0), 3)

        # Regulatory issues are usually rare, so we skew it heavily towards 0
        has_regulatory_issues = round(random.betavariate(1, 5), 3)

        # Logic for recommend_buy (Fuzzy Logic Rule with added Noise)
        # Base logical score using fuzzy logic principles:
        # Profitability is good, debt is bad, regulatory issues are terrible
        # Cash flow and management act as a supporting OR condition

        strength_score = max(has_strong_cash_flow, has_good_management)
        baseline_score = is_profitable * (1.0 - has_high_debt) * (1.0 - has_regulatory_issues)

        # Combine the scores
        raw_buy_score = baseline_score * strength_score

        # Add a bit of market "noise" or unpredictability (approx +/- 0.1)
        noise = random.uniform(-0.1, 0.1)
        final_buy_score = raw_buy_score + noise

        # Threshold the final score to make a binary decision (Buy = 1, Don't Buy = 0)
        # Or we could leave it continuous, but typically targets are binary. We'll threshold at 0.35.
        recommend_buy = 1 if final_buy_score >= 0.35 else 0

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

def save_to_csv(data, filename, include_target=True):
    # Copy data to avoid modifying the original dicts
    data_to_save = [dict(row) for row in data]

    if not include_target:
        for row in data_to_save:
            row.pop("recommend_buy", None)

    keys = data_to_save[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data_to_save)

# Generate datasets (Increasing size slightly to help neural net learn fuzzy rules)
train_data = generate_data(800)
test_data = generate_data(200)

# Save to CSV
save_to_csv(train_data, "train.csv", include_target=True)
save_to_csv(test_data, "test.csv", include_target=False)
save_to_csv(test_data, "solution.csv", include_target=True)

print("Version 2 fuzzy/noisy datasets generated successfully: train.csv, test.csv, and solution.csv.")

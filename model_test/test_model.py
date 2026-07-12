import pandas as pd
import torch
import ltn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using Device: {device.type.upper()}")

# 1. LOAD THE DATASET
df = pd.read_csv("../train.csv")

# Combine the 5 features into a single input tensor for the RecommendBuy model
feature_cols = ['is_profitable', 'has_high_debt', 'has_strong_cash_flow', 'has_good_management', 'has_regulatory_issues']
features = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=device)
labels_buy = torch.tensor(df['recommend_buy'].tolist(), dtype=torch.float32, device=device)

# 2. DEFINE THE NEURAL NETWORK (PREDICATE)
class RecommendBuyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(5, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.layer(x)

model_buy = RecommendBuyModel().to(device)
RecommendBuy = ltn.Predicate(model_buy)

# We define simple predicates that just extract the known feature at the specific index
class ExtractFeature(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        # the model expects a tensor with 5 columns
        return x[:, self.index].unsqueeze(-1)

IsProfitable = ltn.Predicate(ExtractFeature(0))
HasHighDebt = ltn.Predicate(ExtractFeature(1))
HasStrongCashFlow = ltn.Predicate(ExtractFeature(2))
HasGoodManagement = ltn.Predicate(ExtractFeature(3))
HasRegulatoryIssues = ltn.Predicate(ExtractFeature(4))

# 3. SET UP FUZZY LOGIC OPERATORS
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrMax())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")

optimizer = torch.optim.Adam(model_buy.parameters(), lr=0.01)

# 5. TRAINING LOOP
print(f"\n🚀 Starting training on {len(df)} samples using {device.type.upper()}...")
for epoch in range(150):
    optimizer.zero_grad()

    x_all = ltn.Variable("x_all", features)

    # Extract variables based on labels
    x_buy = ltn.Variable("x_buy", features[labels_buy == 1])
    x_not_buy = ltn.Variable("x_not_buy", features[labels_buy == 0])

    # --- DEFINE LOGIC RULES (THE KNOWLEDGE BASE) ---
    has_strength = Or(HasStrongCashFlow(x_all), HasGoodManagement(x_all))
    clean_baseline = And(IsProfitable(x_all), And(Not(HasHighDebt(x_all)), Not(HasRegulatoryIssues(x_all))))
    target_condition = And(clean_baseline, has_strength)

    # Domain Knowledge: Target implies condition & condition implies target
    rule1 = Forall(x_all, Implies(RecommendBuy(x_all), target_condition))
    rule2 = Forall(x_all, Implies(target_condition, RecommendBuy(x_all)))

    # Data Supervision Rules: Learn from the noisy ground-truth labels
    rule3 = Forall(x_buy, RecommendBuy(x_buy))
    rule4 = Forall(x_not_buy, Not(RecommendBuy(x_not_buy)))

    knowledge_base = And(And(rule1, rule2), And(rule3, rule4))

    sat_level = knowledge_base.value
    loss = 1.0 - sat_level

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1:3d} | Satisfaction Level: {sat_level.item():.4f} | Loss: {loss.item():.4f}")

print("\n🎉 Training complete!")

model_buy.eval()

# 6. READ EVALUATION DATASET
df_test = pd.read_csv("../test.csv")
df_sol = pd.read_csv("../solution.csv")
df_eval = pd.merge(df_test, df_sol[['company', 'recommend_buy']], on='company', how='inner')

test_features = torch.tensor(df_eval[feature_cols].values, dtype=torch.float32, device=device)

# 7. RUN INFERENCE ONLY FOR THE TARGET METRIC
with torch.no_grad():
    pred_buy_fuzzy = RecommendBuy.model(test_features).squeeze(-1)

# 8. POST-PROCESS CONTINUOUS FUZZY VALUES INTO BINARY LABELS
df_eval['pred_recommend_buy_prob'] = pred_buy_fuzzy.cpu().numpy()

threshold = 0.5
df_eval['pred_recommend_buy'] = (df_eval['pred_recommend_buy_prob'] >= threshold).astype(int)

print("\n🔮 Evaluation Results (First 10 Rows):")
cols_to_show = [
    'company',
    'is_profitable',
    'has_high_debt',
    'recommend_buy',
    'pred_recommend_buy'
]
print(df_eval[cols_to_show].head(10).to_string(index=False))

acc = (df_eval['recommend_buy'] == df_eval['pred_recommend_buy']).mean() * 100
print(f"\n🎯 Accuracy for {'recommend_buy':<25}: {acc:.2f}%")

import csv

with open("quote_requests_sample.csv", encoding="utf-8-sig") as f:
    inputs = list(csv.DictReader(f))

with open("test_results.csv", encoding="utf-8") as f:
    results = list(csv.DictReader(f))

fulfilled = sum(r["status"] == "fulfilled" for r in results)
cash_changes = sum(float(r["cash_change"]) != 0 for r in results)
unfulfilled = sum(
    r["status"] in {"rejected", "unfulfilled", "clarification"}
    for r in results
)

print("Input requests:", len(inputs))
print("Processed requests:", len(results))
print("Fulfilled orders:", fulfilled)
print("Cash balance changes:", cash_changes)
print("Rejected/unfulfilled requests:", unfulfilled)

assert len(inputs) == len(results), "Not all requests processed"
assert fulfilled >= 3, "Fewer than 3 fulfilled orders"
assert cash_changes >= 3, "Fewer than 3 cash changes"
assert unfulfilled >= 1, "No rejected/unfulfilled requests"

print("All evaluation thresholds passed!")
PY
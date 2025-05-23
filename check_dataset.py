import json

with open('flipped_dataset/dataset_metadata.json', 'r') as f:
    data = json.load(f)

original_counts = {}
flipped_counts = {}

for sample in data['samples']:
    orig = sample['original_label']
    flipped = sample['flipped_label']
    original_counts[orig] = original_counts.get(orig, 0) + 1
    flipped_counts[flipped] = flipped_counts.get(flipped, 0) + 1

print('Original distribution:')
for emotion, count in sorted(original_counts.items()):
    print(f'  {emotion}: {count}')

print('\nFlipped distribution:')
for emotion, count in sorted(flipped_counts.items()):
    print(f'  {emotion}: {count}')

print('\nHappy → Sad flips:')
happy_flips = [s for s in data['samples'] if 'happy' in s['original_label'].lower()]
print(f'Total happy images flipped: {len(happy_flips)}')
for sample in happy_flips[:5]:
    print(f'  {sample["original_label"]} → {sample["flipped_label"]}') 
require './TexteSuisse.rb'

# After your existing setup code:
filepath = 'data.csv'
loader = TextSuisse::DataLoader.new
toponyms = loader.load_toponyms(filepath)

tokenizer = TextSuisse::Tokenizer.new
tokenized, chars, c2i, i2c = tokenizer.tokenize(toponyms)

# 1. Create counting-based model
bigram_generator = TextSuisse::BigramGenerator.new
counting_probs = bigram_generator.generate(tokenized, c2i)

# 2. Create trained model
dataset_creator = TextSuisse::BigramDatasetCreator.new
dataset = dataset_creator.create_dataset(tokenized, c2i)
split_data = dataset_creator.split_dataset(dataset)

model = TextSuisse::BigramModel.new(chars.size)
model.train(split_data[:train])

# 3. Compare them
comparator = TextSuisse::BigramComparator.new(
  tokenizer,
  counting_probs,
  model,
  c2i,
  i2c
)

# Compare probability distributions for a specific character
puts "Probability distribution comparison:"
puts comparator.compare_probability_distributions('a')

# Compare generated samples
puts "\nGeneration comparison:"
comparison = comparator.compare_generated_samples(5)
puts "Counting-based samples: #{comparison[:counting]}"
puts "Trained-model samples: #{comparison[:trained]}"

# Evaluate on test data
puts "\nTest performance:"
perf = comparator.evaluate_models(split_data[:val])
puts "Counting model loss: #{perf[:counting_loss]}"
puts "Trained model loss: #{perf[:trained_loss]}"

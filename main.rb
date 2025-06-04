# File: main.rb
require './TexteSuisse.rb'

filepath = 'data.csv'
csv_loader = TextSuisse::DataLoader.new
toponym_list = csv_loader.load_toponyms(filepath)

tokenizer = TextSuisse::Tokenizer.new
tokenized_toponyms, characters, c2i, i2c = tokenizer.tokenize(toponym_list)

bigram_generator = TextSuisse::BigramGenerator.new
bigram = bigram_generator.generate(tokenized_toponyms, c2i)


generator = TextSuisse::TextGenerator.new(bigram, c2i, i2c)

puts "Generated toponyms:"
5.times { puts generator.generate_word }

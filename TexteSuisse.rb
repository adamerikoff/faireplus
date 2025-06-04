require 'csv'
require 'set'

module TextSuisse
  class Tokenizer
    DEFAULT_TOKEN_MAP = {
      ' ' => '<SPACE>',
      '.' => '<PERIOD>',
      '-' => '<HYPHEN>',
      "'" => '<APOSTROPHE>',
    }.freeze

    def initialize(token_map = DEFAULT_TOKEN_MAP)
      @token_map = token_map
    end

    def tokenize_word(word)
      return [] if word.nil? || word.empty?
      word.strip.downcase.chars.map do |char|
        @token_map.fetch(char, char)
      end
    end

    def char2index(character_set)
      character_set.each_with_index.to_h
    end

    def index2char(character_set)
      character_set.each_with_index.to_h { |string, index| [index, string] }
    end

    def tokenize(words)
      character_set = Set.new
      tokenized_words = words.map do |word|
        tokens = ["<S>"] + tokenize_word(word) + ["<E>"]
        tokens.each { |token| character_set.add(token) }
        tokens
      end
      characters = character_set.sort
      char2index = self.char2index(characters)
      index2char = self.index2char(characters)
      [tokenized_words, characters, char2index, index2char]
    end
  end

  class BigramGenerator
    def generate(tokenized_words, char2index)
      size = char2index.size
      bigrams = Array.new(size) { Array.new(size, 1) }
      tokenized_words.each do |tokenized_word|
          tokenized_word.each_cons(2) do |token1, token2|
          i = char2index[token1]
          j = char2index[token2]
          bigrams[i][j] += 1
          end
      end
      probabilities = bigrams.map do |row|
        row_sum = row.sum.to_f
        row_sum.zero? ? Array.new(row.size, 0.0) : row.map { |count| count / row_sum }
      end
      probabilities
    end
  end

  class BigramDatasetCreator
    def create_dataset(tokenized_words, char2index)
      dataset = []
      tokenized_words.each do |tokenized_word|
        tokenized_word.each_cons(2) do |token1, token2|
          dataset << {
            input: char2index[token1],
            target: char2index[token2]
          }
        end
      end
      dataset
    end

    def split_dataset(dataset, train_ratio = 0.8)
      shuffled = dataset.shuffle
      train_size = (dataset.size * train_ratio).to_i
      {
        train: shuffled[0...train_size],
        val: shuffled[train_size..-1]
      }
    end
  end

  class BigramModel
    def initialize(vocab_size)
      @vocab_size = vocab_size
      @weights = Array.new(vocab_size) { Array.new(vocab_size, 0.0) }
    end

    def train(train_data, learning_rate: 0.01, epochs: 50)
      epochs.times do |epoch|
        train_data.each do |example|
          input, target = example[:input], example[:target]

          # Forward pass
          probs = softmax(@weights[input])

          # Compute loss (cross-entropy)
          loss = -Math.log(probs[target])

          # Backward pass
          # Gradient of cross-entropy loss with softmax
          grad = probs.dup
          grad[target] -= 1.0

          # Update weights
          @weights[input].each_with_index do |_, j|
            @weights[input][j] -= learning_rate * grad[j]
          end
        end

        puts "Epoch #{epoch + 1}, Loss: #{average_loss(train_data)}" if epoch % 1 == 0
      end
    end

    def average_loss(data)
      total = 0.0
      data.each do |example|
        input, target = example[:input], example[:target]
        probs = softmax(@weights[input])
        total += -Math.log(probs[target])
      end
      total / data.size
    end

    def softmax(logits)
      max_logit = logits.max
      exps = logits.map { |x| Math.exp(x - max_logit) }
      sum_exps = exps.sum
      exps.map { |exp| exp / sum_exps }
    end

    def predict_probs(input_index)
      softmax(@weights[input_index])
    end
  end

  class DataLoader
    def load_toponyms(filepath)
      toponyms = []
      CSV.foreach(filepath, headers: false) do |row|
        toponyms << row[0] if row[0]
      end
      toponyms
    end
  end

  class TextGenerator
    REVERSE_TOKEN_MAP = {
      '<SPACE>' => ' ',
      '<PERIOD>' => '.',
      '<HYPHEN>' => '-',
      '<APOSTROPHE>' => "'"
    }.freeze

    def initialize(bigram_probs, char_to_index, index_to_char)
      @bigram = bigram_probs
      @c2i = char_to_index
      @i2c = index_to_char
    end

    def generate_word(max_length = 20)
      current_char = '<S>'
      generated = []

      loop do
        probs = @bigram[@c2i[current_char]]
        next_char_idx = sample_from_probs(probs)
        next_char = @i2c[next_char_idx]

        break if next_char == '<E>' || generated.size >= max_length

        unless ['<S>', '<E>'].include?(next_char)
          # Replace special tokens with real characters
          char_to_add = REVERSE_TOKEN_MAP.fetch(next_char, next_char)
          generated << char_to_add
        end

        current_char = next_char
      end

      generated.join
    end

    private
    def sample_from_probs(probs)
      r = rand
      cumulative = 0.0

      probs.each_with_index do |p, i|
        cumulative += p
        return i if r <= cumulative
      end
      probs.each_with_index.max_by { |p, _| p }[1]
    end

  end

  class BigramComparator
    def initialize(tokenizer, counting_model, trained_model, char_to_index, index_to_char)
      @tokenizer = tokenizer
      @counting_model = counting_model  # From BigramGenerator
      @trained_model = trained_model    # From BigramModel
      @c2i = char_to_index
      @i2c = index_to_char
    end

    # 1. Compare Probability Distributions
    def compare_probability_distributions(input_char)
      input_idx = @c2i[input_char]

      counting_probs = @counting_model[input_idx]
      trained_probs = @trained_model.predict_probs(input_idx)

      { counting: counting_probs,
        trained: trained_probs,
        kl_divergence: kl_divergence(counting_probs, trained_probs) }
    end

    # 2. Compare Generation Quality
    def compare_generated_samples(num_samples=5)
      counting_gen = TextGenerator.new(@counting_model, @c2i, @i2c)
      trained_gen = TextGenerator.new(@trained_model.method(:predict_probs), @c2i, @i2c)

      { counting: num_samples.times.map { counting_gen.generate_word },
        trained: num_samples.times.map { trained_gen.generate_word } }
    end

    # 3. Compare Predictive Performance
    def evaluate_models(test_data)
      counting_loss = 0.0
      trained_loss = 0.0

      test_data.each do |example|
        input_idx = example[:input]
        target_idx = example[:target]

        # Counting model evaluation
        counting_probs = @counting_model[input_idx]
        counting_loss += -Math.log(counting_probs[target_idx])

        # Trained model evaluation
        trained_probs = @trained_model.predict_probs(input_idx)
        trained_loss += -Math.log(trained_probs[target_idx])
      end

      { counting_loss: counting_loss / test_data.size,
        trained_loss: trained_loss / test_data.size }
    end

    private

    def kl_divergence(p, q)
      p.each_with_index.sum do |p_val, i|
        p_val > 0 ? p_val * Math.log(p_val / q[i]) : 0
      end
    end
  end
end

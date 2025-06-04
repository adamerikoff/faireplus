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
      bigrams = Array.new(size) { Array.new(size, 0) }
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

        generated << next_char unless ['<S>', '<E>', '<SPACE>'].include?(next_char)
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
end

# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tokenization classes for python tokenizers. For fast tokenizers (provided by HuggingFace's tokenizers library) see
tokenization_utils_fast.py
"""

import bisect
import itertools
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, overload

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


class Trie:
  """
  Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
  Loose reference https://en.wikipedia.org/wiki/Trie
  """

  def __init__(self, *args):
    self.data = {}
    self._tokens = set()
    self._termination_char = ""
    self.update(*args)

  def update(self, *args):
    """
    Updates the Trie with new tokens provided as arguments.

    Args:
      *args: Variable number of words to be added to the Trie.
    """
    for token in tuple(*args):
      self.add(token)

  def add(self, word: str):
    """
    Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
    The special key `""` in `self._termination_char` is used to represent termination.

    This function is idempotent, adding twice the same word will leave the trie unchanged

    Example:

    ```python
    >>> trie = Trie()
    >>> trie.add("Hello 友達")
    >>> trie.data
    {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

    >>> trie.add("Hello")
    >>> trie.data
    {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
    ```
    """
    if not word:
      # Prevent empty string
      return

    self._tokens.add(word)
    ref = self.data
    for char in word:
      ref[char] = ref.setdefault(char, {})
      ref = ref[char]
    ref[self._termination_char] = 1

  def split(self, text: str) -> List[str]:
    """
    Will look for the words added to the trie within `text`. Output is the original string splitted along the
    boundaries of the words found.

    This trie will match the longest possible word first !

    Example:

    ```python
    >>> trie = Trie()
    >>> trie.split("[CLS] This is a extra_id_100")
    ["[CLS] This is a extra_id_100"]

    >>> trie.add("[CLS]")
    >>> trie.add("extra_id_1")
    >>> trie.add("extra_id_100")
    >>> trie.split("[CLS] This is a extra_id_100")
    ["[CLS]", " This is a ", "extra_id_100"]
    ```
    """
    # indexes are counted left of the chars index.
    # "hello", index 0, is left of h, index 1 is between h and e.
    # index 5 is right of the "o".

    # States are going to capture every possible start (indexes as above)
    # as keys, and have as values, a pointer to the position in the trie
    # where we're at. This is a partial match for now.
    # This enables to keep track of multiple matches while we're iterating
    # the string
    # If the trie contains, "blowing", and "lower" and we encounter the
    # string "blower", we need to split into ["b", "lower"].
    # This is where we need to keep track of multiple possible starts.
    states = OrderedDict()

    # This will contain every indices where we need
    # to cut.
    # We force to cut at offset 0 and len(text) (added later)
    offsets = [0]

    # This is used by the lookahead which needs to skip over
    # some text where the full match exceeded the place in the initial
    # for loop
    skip = 0
    # Main loop, Giving this algorithm O(n) complexity
    for current, current_char in enumerate(text):
      if skip and current < skip:
        # Prevents the lookahead for matching twice
        # like extra_id_100 and id_100
        continue

      # This will track every state
      # that stop matching, we need to stop tracking them.
      # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
      # fail on "b", we need to remove 0 from the valid states.
      to_remove = set()
      # Whenever we found a match, we need to drop everything
      # this is a greedy algorithm, it will match on the first found token
      reset = False

      # In this case, we already have partial matches (But unfinished)
      for start, trie_pointer in states.items():
        if "" in trie_pointer:
          # This is a final match, we need to reset and
          # store the results in `offsets`.

          # Lookahead to match longest first
          # Important in case of extra_id_1 vs extra_id_100
          # Here we are also actively looking for other earlier partial
          # matches
          # "[CLS]", "L", we need to match CLS even if L is special
          for lookstart, looktrie_pointer in states.items():
            if lookstart > start:
              # This partial match is later, we can stop looking
              break
            elif lookstart < start:
              # This partial match is earlier, the trie pointer
              # was already updated, so index is + 1
              lookahead_index = current + 1
              end = current + 1
            else:
              # Here lookstart == start and
              #      looktrie_pointer == trie_pointer
              # It wasn't updated yet so indices are current ones
              lookahead_index = current
              end = current
            next_char = text[lookahead_index] if lookahead_index < len(text) else None
            if "" in looktrie_pointer:
              start = lookstart
              end = lookahead_index
              skip = lookahead_index

            while next_char in looktrie_pointer:
              looktrie_pointer = looktrie_pointer[next_char]
              lookahead_index += 1
              if "" in looktrie_pointer:
                start = lookstart
                end = lookahead_index
                skip = lookahead_index

              if lookahead_index == len(text):
                # End of string
                break
              next_char = text[lookahead_index]
            # End lookahead

          # Storing and resetting
          offsets.append(start)
          offsets.append(end)
          reset = True
          break
        elif current_char in trie_pointer:
          # The current character being looked at has a match within the trie
          # update the pointer (it will be stored back into states later).
          trie_pointer = trie_pointer[current_char]

          # Storing back the new pointer into the states.
          # Partial matches got longer by one.
          states[start] = trie_pointer
        else:
          # The new character has not match in the trie, we need
          # to stop keeping track of this partial match.
          # We can't do it directly within the loop because of how
          # python iteration works
          to_remove.add(start)

      # Either clearing the full start (we found a real match)
      # Or clearing only the partial matches that didn't work.
      if reset:
        states = {}
      else:
        for start in to_remove:
          del states[start]

      # If this character is a starting character within the trie
      # start keeping track of this partial match.
      if current >= skip and current_char in self.data:
        states[current] = self.data[current_char]

    # We have a cut at the end with states.
    for start, trie_pointer in states.items():
      if "" in trie_pointer:
        # This is a final match, we need to reset and
        # store the results in `offsets`.
        end = len(text)
        offsets.append(start)
        offsets.append(end)
        # Longest cut is always the one with lower start so the first
        # item so we need to break.
        break

    return self.cut_text(text, offsets)

  def cut_text(self, text, offsets):
    # We have all the offsets now, we just need to do the actual splitting.
    # We need to eventually add the first part of the string and the eventual
    # last part.
    offsets.append(len(text))
    tokens = []
    start = 0
    for end in offsets:
      if start > end:
        logger.error(
          "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
          " anyway."
        )
        continue
      elif start == end:
        # This might happen if there's a match at index 0
        # we're also preventing zero-width cuts in case of two
        # consecutive matches
        continue
      tokens.append(text[start:end])
      start = end

    return tokens


class ExtensionsTrie(Trie):
  def __init__(self, *args):
    super().__init__(*args)

  def extensions(self, prefix: str):
    """
    Generates all extensions of a given prefix token in the Trie.

    Example:

    ```python
    >>> trie = Trie()
    >>> trie.add("apple")
    >>> trie.add("app")
    >>> trie.add("application")
    >>> trie.extensions("app")
    ['app', 'apple', 'application']
    ```
    """
    prefix_node = self._get_node(prefix)
    ret = self._collect_tokens(prefix_node)
    return [prefix + token for token in ret]

  def _get_node(self, token: str) -> dict:
    """
    Retrieves the node corresponding to the given token in the Trie.

    Args:
      token (str): The token for which the corresponding node needs to be retrieved.

    Returns:
      dict: The node in the Trie corresponding to the given token.
    """
    node = self.data
    for char in token:
      if char not in node:
        break

      node = node[char]
    return node

  def _collect_tokens(self, node: dict) -> list:
    """
    Generates all tokens in the Trie starting from a given node.

    Args:
      node (dict): The node in the Trie from which tokens need to be generated.

    Returns:
      list: List of tokens generated from the given node.
    """
    tokens = [self._termination_char] if self._termination_char in node else []
    for token, subtrie_head in node.items():
      if token != self._termination_char:
        subtokens = self._collect_tokens(subtrie_head)
        tokens.extend([token + subtoken for subtoken in subtokens])
    return tokens


def _is_whitespace(char):
  """Checks whether `char` is a whitespace character."""
  # \t, \n, and \r are technically control characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `char` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `char` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def _is_end_of_word(text):
  """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
  last_char = text[-1]
  return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
  """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
  first_char = text[0]
  return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
  """
  Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
  """
  insertion_idx = bisect.bisect_left(token_list, new_token)
  # Checks if new_token is already in the ordered token_list
  if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
    # new_token is in token_list, don't add
    return
  else:
    token_list.insert(insertion_idx, new_token)


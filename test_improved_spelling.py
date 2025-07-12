#!/usr/bin/env python3
"""
Test the improved spelling correction functionality.
"""

from difflib import get_close_matches

def test_improved_spelling_correction():
    """Test the improved spelling correction logic."""
    
    # Sample vocabulary (words that should be in the index)
    vocabulary = {
        'sky', 'blue', 'color', 'star', 'cloud', 'bright', 'day', 'night',
        'dark', 'light', 'sun', 'moon', 'weather', 'clear', 'beautiful', 'amazing',
        'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
    }
    
    # Common stop words that should rarely be corrected
    common_stop_words = {'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def correct_spelling_improved(query_tokens, confidence_threshold=0.8):
        """Improved spelling correction logic."""
        print(f"Correcting spelling for query: {' '.join(query_tokens)}")
        
        corrected_tokens = []
        corrections_made = []
        
        for token in query_tokens:
            # If token exists in vocabulary, keep it as is
            if token in vocabulary:
                corrected_tokens.append(token)
                continue
            
            # Be very conservative with common words
            if token.lower() in common_stop_words:
                corrected_tokens.append(token)
                print(f"Keeping common word '{token}' as is")
                continue
            
            # Find the closest match in vocabulary
            matches = get_close_matches(token, vocabulary, n=1, cutoff=confidence_threshold)
            
            if matches:
                corrected_token = matches[0]
                # Additional check: only correct if the correction is significantly different
                # and the original token is clearly misspelled
                if len(token) > 2 and len(corrected_token) > 2:
                    # Check if the correction makes sense
                    if corrected_token.lower() != token.lower():
                        corrected_tokens.append(corrected_token)
                        corrections_made.append((token, corrected_token))
                        print(f"Corrected '{token}' to '{corrected_token}'")
                    else:
                        corrected_tokens.append(token)
                        print(f"Keeping '{token}' as is (correction too similar)")
                else:
                    corrected_tokens.append(token)
                    print(f"Keeping short token '{token}' as is")
            else:
                # If no good match found, keep the original token
                corrected_tokens.append(token)
                print(f"No correction found for '{token}', keeping original")
        
        if corrections_made:
            print(f"Spelling corrections made: {corrections_made}")
        else:
            print("No spelling corrections needed.")
            
        return corrected_tokens
    
    # Test cases
    test_cases = [
        "whyyy is the sku blu",  # Should correct 'sku' to 'sky' and 'blu' to 'blue', but keep 'is' and 'the'
        "sky is blue",            # Should keep everything as is
        "thee skky is bluu",      # Should correct 'skky' to 'sky' and 'bluu' to 'blue', but keep 'thee' as is
        "zis zthe sky blue",      # Should keep 'zis' and 'zthe' as is (they're not in vocabulary)
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Original: '{test_query}'")
        
        tokens = test_query.split()
        corrected_tokens = correct_spelling_improved(tokens)
        corrected_query = ' '.join(corrected_tokens)
        
        print(f"Result: '{corrected_query}'")
        print("-" * 50)

if __name__ == "__main__":
    test_improved_spelling_correction() 
"""
Node A: Semantic Router
Embeds static merchant data and caller utterances for semantic matching.
"""
import asyncio
from sentence_transformers import SentenceTransformer, util

print("Loading embedding model for Node A...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Comprehensive Merchant Data + Conversational Intents
MENU_DATA = [
    # --- Combo Meals ---
    {"id": "C_02", "type": "ITEM", "name": "5 Tenders Combo", "text": "5 Chicken Tenders Combo meal with fries and drink."},
    
    {"id": "C_01", "type": "ITEM", "name": "Chicken Sandwich Combo", "text": "Chicken Sandwich Combo meal. Comes with a sandwich, 1 fry, and a 20 oz soda."},
    {"id": "C_05", "type": "ITEM", "name": "Family Combo", "text": "Family Combo. Includes exactly 10 Traditional Wings, 10 Boneless Wings, 5 Tenders, Large Fry, and a 2 Liter Drink."},
    {"id": "C_06", "type": "ITEM", "name": "Nashville Hot Chicken Sandwich", "text": "Nashville Hot Chicken Sandwich. Hand-breaded chicken breast with Cajun Rub, Awesome Aioli, and pickles on a brioche bun."},
    {"id": "C_03", "type": "ITEM", "name": "5 Boneless Wings Combo", "text": "5 Boneless Wings Combo meal with fries and soda."},
    {"id": "C_04", "type": "ITEM", "name": "10 Traditional Wings Combo", "text": "10 Traditional Bone-In Wings Combo meal with fries and drink."},
    
    
    # --- Platters & Pizza ---
    {"id": "P_01", "type": "ITEM", "name": "Flat Bread Pizza", "text": "Flat Bread Pizza."},
    {"id": "PL_01", "type": "ITEM", "name": "Chicken Over Rice", "text": "Chicken Over Rice Platter."},
    {"id": "PL_02", "type": "ITEM", "name": "Gyro Over Rice", "text": "Gyro Over Rice Platter."},
    
    # --- Sides & Salads ---
    {"id": "S_01", "type": "ITEM", "name": "Waffle Fries", "text": "Famous Waffle Fries."},
    {"id": "S_02", "type": "ITEM", "name": "Mac & Cheese", "text": "Mac & Cheese side."},
    {"id": "S_03", "type": "ITEM", "name": "Mozzarella Sticks", "text": "Mozzarella Sticks."},
    {"id": "S_04", "type": "ITEM", "name": "Jalapeño Poppers", "text": "Jalapeno Poppers."},
    {"id": "SL_01", "type": "ITEM", "name": "Caesar Salad", "text": "Caesar Salad."},

    # --- Modifiers ---
    {"id": "MOD_01", "type": "MODIFIER", "name": "No Cheese", "text": "Modifier: No cheese, without cheese, take off cheese."},
    {"id": "MOD_02", "type": "MODIFIER", "name": "Add Ranch", "text": "Modifier: Add Ranch dip or Blue Cheese side."},
    {"id": "MOD_03", "type": "MODIFIER", "name": "Extra Sauce", "text": "Modifier: Extra wing sauce, aioli, spicy sauce."},

    # --- Conversational Intents ---
    {"id": "INT_GREETING", "type": "INTENT", "name": "Greeting", "text": "Hi, hello, hey there, good morning, my name is, this is."},
    {"id": "INT_CHECK_CART", "type": "INTENT", "name": "Check Cart", "text": "What is in my cart? What did I order? My shopping list, check my items, what do I have?"},
    {"id": "INT_INQUIRE_MENU", "type": "INTENT", "name": "Menu Inquiry", "text": "What do you have? What are you meant to help me with? What is on the menu? What can I order? I need a combo. What combos do you have? What meals do you sell?"},
    {"id": "INT_CHECKOUT", "type": "INTENT", "name": "Checkout", "text": "Checkout, pay, finish the order, ready for total, that is all, that would be it, that's it, I'm done."}
]

semantic_strings = [item["text"] for item in MENU_DATA]
print("Embedding static menu & intent vectors...\n")
menu_embeddings = model.encode(semantic_strings, convert_to_tensor=True)

def get_item_name(item_id: str) -> str:
    for item in MENU_DATA:
        if item["id"] == item_id:
            return item["name"]
    return item_id

async def match_by_meaning(user_text: str, threshold: float = 0.40):
    def _compute_match():
        utterance_embedding = model.encode(user_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(utterance_embedding, menu_embeddings)[0]
        best_match_idx = cosine_scores.argmax().item()
        best_score = cosine_scores[best_match_idx].item()
        
        if best_score < threshold:
            return {"id": "LOW_CONF", "type": "LOW_CONFIDENCE", "name": "Unknown"}, best_score
            
        return MENU_DATA[best_match_idx], best_score

    return await asyncio.to_thread(_compute_match)
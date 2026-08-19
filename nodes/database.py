"""
Node C Sub-module: SQLite Database
Handles pricing lookups and cart totals using raw SQL.
"""
import sqlite3

def init_db():
    conn = sqlite3.connect('menu.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS menu (item_id TEXT PRIMARY KEY, price REAL)''')
    
    prices = [
        ("C_01", 14.19), ("C_02", 17.19), ("C_03", 14.19), ("C_04", 20.19), ("C_05", 42.99),
        ("P_01", 11.99), ("PL_01", 10.99), ("PL_02", 11.99),
        ("S_01", 3.99),  ("S_02", 4.99),  ("S_03", 5.99), ("S_04", 4.99), ("SL_01", 5.99),
        # Modifiers
        ("MOD_01", 0.00), # No Cheese is free
        ("MOD_02", 1.49), # Ranch / Blue Cheese
        ("MOD_03", 1.49)  # Any Wing Sauce
    ]
    cursor.executemany('INSERT OR IGNORE INTO menu VALUES (?,?)', prices)
    conn.commit()
    conn.close()

def get_item_price(item_id):
    """Fetches the price of a single item."""
    conn = sqlite3.connect('menu.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price FROM menu WHERE item_id = ?', (item_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0.00

def calculate_total(cart_ids):
    total = sum(get_item_price(item_id) for item_id in cart_ids)
    return total
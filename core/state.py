import json
from fakeredis import FakeAsyncRedis

# Initialize the mock async Redis client directly in memory
redis_client = FakeAsyncRedis(decode_responses=True)

class SessionManager:
    def __init__(self, session_id):
        self.session_id = f"session:{session_id}"

    async def initialize_call(self):
        """Set the starting state when the phone rings."""
        await redis_client.hset(self.session_id, mapping={
            "state": "GREETING",
            "cart": json.dumps([]) # Physical cart to hold item IDs
        })

    async def get_state(self):
        return await redis_client.hget(self.session_id, "state")

    async def get_cart(self):
        cart_data = await redis_client.hget(self.session_id, "cart")
        return json.loads(cart_data) if cart_data else []

    async def update_state(self, new_state):
        await redis_client.hset(self.session_id, "state", new_state)
        
    async def add_to_cart(self, item_id):
        """Adds a menu_id to the cart. Allows multiple occurrences of the same item."""
        cart = await self.get_cart()
        cart.append(item_id)
        await redis_client.hset(self.session_id, "cart", json.dumps(cart))
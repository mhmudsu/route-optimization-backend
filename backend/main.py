from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import googlemaps
from scipy.spatial.distance import cdist
import numpy as np

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Route Optimization API",
    description="Backend for optimizing delivery routes",
    version="1.0.0"
)

# CORS - allow Retool to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify Retool domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Maps client
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else None

# Pydantic models for request/response
class Order(BaseModel):
    order_id: str
    customer_name: str
    address: str
    weight_kg: float
    priority: int

class OptimizeRouteRequest(BaseModel):
    orders: List[Order]
    start_address: Optional[str] = "Amsterdam, Netherlands"

class OptimizedOrder(BaseModel):
    order_id: str
    customer_name: str
    address: str
    weight_kg: float
    priority: int
    sequence: int
    estimated_arrival: Optional[str] = None
    distance_from_previous_km: Optional[float] = None

class OptimizeRouteResponse(BaseModel):
    optimized_orders: List[OptimizedOrder]
    total_distance_km: float
    total_time_minutes: float
    fuel_cost_estimate_eur: float

# Health check endpoint
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Route Optimization API",
        "version": "1.0.0",
        "google_maps_configured": gmaps is not None
    }

@app.get("/api/health")
def api_health():
    return health_check()

# Helper function: Geocode address to lat/lng
def geocode_address(address: str) -> tuple:
    """Convert address to (lat, lng) coordinates"""
    if not gmaps:
        # Fallback to dummy coordinates if no API key
        # In real scenario, this would fail
        return (52.3676, 4.9041)  # Amsterdam
    
    try:
        result = gmaps.geocode(address)
        if result:
            location = result[0]['geometry']['location']
            return (location['lat'], location['lng'])
        else:
            raise ValueError(f"Could not geocode address: {address}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geocoding error: {str(e)}")

# Helper function: Calculate distance matrix
def calculate_distance_matrix(coordinates: List[tuple]) -> np.ndarray:
    """Calculate distance matrix between all coordinate pairs"""
    # Use Euclidean distance as approximation
    # In production, use Google Distance Matrix API for real road distances
    coords_array = np.array(coordinates)
    distances = cdist(coords_array, coords_array, metric='euclidean')
    
    # Convert to kilometers (rough approximation: 1 degree ≈ 111 km)
    distances_km = distances * 111
    
    return distances_km

# Helper function: Solve TSP with priority weights
def solve_tsp_with_priority(distance_matrix: np.ndarray, priorities: List[int], start_index: int = 0) -> List[int]:
    """
    Solve Traveling Salesman Problem with priority consideration
    Uses nearest neighbor heuristic with priority weighting
    """
    n = len(distance_matrix)
    unvisited = set(range(n))
    unvisited.remove(start_index)
    
    route = [start_index]
    current = start_index
    
    while unvisited:
        # Calculate weighted scores for unvisited nodes
        scores = []
        for node in unvisited:
            distance = distance_matrix[current][node]
            priority_weight = priorities[node]
            
            # Lower distance = better, Higher priority = better
            # Priority 1 = highest, so invert: 1/priority
            score = distance * priority_weight  # Lower score = better
            scores.append((score, node))
        
        # Pick node with lowest score
        scores.sort()
        next_node = scores[0][1]
        
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    
    return route

# Main route optimization endpoint
@app.post("/api/optimize-route", response_model=OptimizeRouteResponse)
def optimize_route(request: OptimizeRouteRequest):
    """
    Optimize delivery route based on:
    - Addresses (geocoded to coordinates)
    - Priorities (1 = highest)
    - Distances between locations
    """
    
    orders = request.orders
    start_address = request.start_address
    
    if not orders:
        raise HTTPException(status_code=400, detail="No orders provided")
    
    # Step 1: Geocode all addresses
    print(f"Geocoding {len(orders)} addresses...")
    addresses = [start_address] + [order.address for order in orders]
    
    try:
        coordinates = [geocode_address(addr) for addr in addresses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")
    
    # Step 2: Calculate distance matrix
    print("Calculating distances...")
    distance_matrix = calculate_distance_matrix(coordinates)
    
    # Step 3: Solve TSP with priorities
    print("Optimizing route...")
    priorities = [1] + [order.priority for order in orders]  # Start has priority 1
    route_indices = solve_tsp_with_priority(distance_matrix, priorities, start_index=0)
    
    # Remove start location from route (index 0)
    route_indices = route_indices[1:]
    
    # Step 4: Calculate metrics
    total_distance = 0
    optimized_orders = []
    
    for seq, idx in enumerate(route_indices, start=1):
        order_idx = idx - 1  # Adjust for start location offset
        order = orders[order_idx]
        
        # Calculate distance from previous location
        if seq == 1:
            prev_idx = 0  # From start
        else:
            prev_idx = route_indices[seq - 2]
        
        distance_from_prev = distance_matrix[prev_idx][idx]
        total_distance += distance_from_prev
        
        optimized_orders.append(OptimizedOrder(
            order_id=order.order_id,
            customer_name=order.customer_name,
            address=order.address,
            weight_kg=order.weight_kg,
            priority=order.priority,
            sequence=seq,
            distance_from_previous_km=round(distance_from_prev, 2)
        ))
    
    # Calculate estimates
    # Assume average speed: 50 km/h in city
    total_time_minutes = (total_distance / 50) * 60
    
    # Fuel cost estimate: 1 liter per 10 km, €2 per liter
    fuel_liters = total_distance / 10
    fuel_cost = fuel_liters * 2
    
    return OptimizeRouteResponse(
        optimized_orders=optimized_orders,
        total_distance_km=round(total_distance, 2),
        total_time_minutes=round(total_time_minutes, 0),
        fuel_cost_estimate_eur=round(fuel_cost, 2)
    )

# Additional endpoint: Just geocode addresses (for testing)
@app.post("/api/geocode")
def geocode_addresses(addresses: List[str]):
    """Geocode a list of addresses"""
    try:
        results = []
        for addr in addresses:
            coords = geocode_address(addr)
            results.append({
                "address": addr,
                "latitude": coords[0],
                "longitude": coords[1]
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
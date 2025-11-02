from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import requests
from scipy.spatial.distance import cdist
import numpy as np

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Route Optimization API",
    description="Backend for optimizing delivery routes using HERE Platform",
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

# HERE Platform API configuration
HERE_API_KEY = os.getenv("HERE_API_KEY", "")
HERE_GEOCODE_URL = "https://geocode.search.hereapi.com/v1/geocode"
HERE_ROUTE_URL = "https://router.hereapi.com/v8/routes"

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
        "service": "Route Optimization API (HERE Platform)",
        "version": "1.0.0",
        "here_api_configured": bool(HERE_API_KEY)
    }

@app.get("/api/health")
def api_health():
    return health_check()

# Helper function: Geocode address using HERE API
def geocode_address(address: str) -> tuple:
    """Convert address to (lat, lng) coordinates using HERE Geocoding API"""
    if not HERE_API_KEY:
        # Fallback to dummy coordinates if no API key
        print("Warning: No HERE API key, using dummy coordinates")
        return (52.3676, 4.9041)  # Amsterdam
    
    try:
        params = {
            'q': address,
            'apiKey': HERE_API_KEY,
            'limit': 1
        }
        
        response = requests.get(HERE_GEOCODE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('items'):
            position = data['items'][0]['position']
            return (position['lat'], position['lng'])
        else:
            raise ValueError(f"Could not geocode address: {address}")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geocoding error: {str(e)}")

# Helper function: Calculate distance matrix
def calculate_distance_matrix(coordinates: List[tuple]) -> np.ndarray:
    """Calculate distance matrix between all coordinate pairs"""
    # Use Euclidean distance as approximation
    # In production, could use HERE Matrix Routing API for real road distances
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
    
    Lower priority number = higher importance (1 = highest priority)
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
            
            # Priority 1 = highest (should visit first)
            # So multiply distance by priority: higher priority (lower number) = lower score
            score = distance * priority_weight
            scores.append((score, node))
        
        # Pick node with lowest score (closest + highest priority)
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
    - Addresses (geocoded to coordinates using HERE)
    - Priorities (1 = highest)
    - Distances between locations
    """
    
    orders = request.orders
    start_address = request.start_address
    
    if not orders:
        raise HTTPException(status_code=400, detail="No orders provided")
    
    if not HERE_API_KEY:
        raise HTTPException(status_code=500, detail="HERE API key not configured")
    
    # Step 1: Geocode all addresses
    print(f"Geocoding {len(orders)} addresses using HERE API...")
    addresses = [start_address] + [order.address for order in orders]
    
    try:
        coordinates = [geocode_address(addr) for addr in addresses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")
    
    # Step 2: Calculate distance matrix
    print("Calculating distances...")
    distance_matrix = calculate_distance_matrix(coordinates)
    
    # Step 3: Solve TSP with priorities
    print("Optimizing route with priority weighting...")
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
    """Geocode a list of addresses using HERE API"""
    if not HERE_API_KEY:
        raise HTTPException(status_code=500, detail="HERE API key not configured")
    
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
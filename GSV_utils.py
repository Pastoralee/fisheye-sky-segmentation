import numpy as np
import requests
import os
from PIL import Image
from global_land_mask import globe
import random

def download_tile(panoid, x, y, zoom, outdir):
    """
    Download a single image tile from Google Street View.

    Args:
        panoid (str): The panorama ID.
        x (int): Tile x-coordinate.
        y (int): Tile y-coordinate.
        zoom (int): Zoom level.
        outdir (str): Output directory for the tile.

    Returns:
        str or None: Path to the downloaded tile, or None if download failed.
    """
    url = f"https://geo0.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&panoid={panoid}&output=tile&x={x}&y={y}&zoom={zoom}&nbt&fover=2"
    outfile = os.path.join(outdir, f"{x}_{y}.jpg")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(outfile, "wb") as f:
                f.write(response.content)
            return outfile
    except requests.RequestException:
        return None
    return None

def assemble_mosaic(panoid, outdir, tilesize=512, numtilesx=4, numtilesy=2):
    """
    Assemble a mosaic image from downloaded tiles.

    Args:
        panoid (str): The panorama ID.
        outdir (str): Output directory for the mosaic.
        tilesize (int, optional): Size of each tile in pixels. Default is 512.
        numtilesx (int, optional): Number of tiles along the x-axis. Default is 4.
        numtilesy (int, optional): Number of tiles along the y-axis. Default is 2.

    Returns:
        str or None: Path to the saved mosaic image, or None if assembly failed.
    """
    mosaic = Image.new("RGB", (tilesize * numtilesx, tilesize * numtilesy), "black")
    
    for x in range(numtilesx):
        for y in range(numtilesy):
            tile_path = download_tile(panoid, x, y, 2, outdir)
            if tile_path is None:
                return None
            img = Image.open(tile_path)
            mosaic.paste(img, (x * tilesize, y * tilesize))
    
    mosaic_path = os.path.join(outdir, "mosaic.png")
    mosaic.save(mosaic_path)
    return mosaic_path

def equirectangular_to_fisheye(infile, outfile, fisheyesize=1024):
    """
    Convert an equirectangular image to a fisheye projection using equidistant projection.

    Args:
        infile (str): Path to the input equirectangular image.
        outfile (str): Path to save the fisheye image.
        fisheyesize (int, optional): Output fisheye image size. Default is 1024.
    """
    img = Image.open(infile)
    width, height = img.size
    img = img.crop((0, 0, width, height // 2))
    width, height = img.size

    img_np = np.asarray(img)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]  # Remove alpha if present

    # Prepare fisheye image array
    fisheye = np.zeros((fisheyesize, fisheyesize, 3), dtype=np.uint8)

    # Normalized grid [-1, 1]
    x = np.linspace(-1, 1, fisheyesize)
    y = np.linspace(-1, 1, fisheyesize)
    x, y = np.meshgrid(x, y)

    r = np.sqrt(x**2 + y**2)  # radial distance from center
    theta = r * (np.pi / 2)   # equidistant projection (theta = r * π/2)
    phi = np.arctan2(y, x)

    # Convert spherical coordinates to latitude and longitude
    lat = theta               # polar angle from zenith
    lon = phi % (2 * np.pi)   # azimuthal angle [0, 2π]

    # Normalize to image coordinates
    srcx = (lon / (2 * np.pi)) * (width - 1)
    srcy = (lat / (np.pi / 2)) * (height - 1)

    # Clip coordinates safely
    srcx = np.clip(srcx, 0, width - 1).astype(int)
    srcy = np.clip(srcy, 0, height - 1).astype(int)

    # Assign RGB values
    fisheye[:, :, :] = img_np[srcy, srcx]

    # Mask outside unit circle (non-fisheye region)
    fisheye[r > 1] = [0, 0, 0]

    # Save output
    Image.fromarray(fisheye).save(outfile)

def get_panoid(lat, lon, api_key):
    """
    Retrieve the panorama ID for a given latitude and longitude using the Google Street View API.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        api_key (str): Google API key.

    Returns:
        str or None: The panorama ID if found, else None.
    """
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        data = response.json()
        if 'pano_id' in data:
            return data['pano_id']
        else:
            return None
    else:
        print(f"Erreur avec l'API Google Street View : {response.status_code}")
        return None

def equirectangular_to_equisolid(infile, outfile, fisheyesize=1024):
    """
    Convert an equirectangular image to an equisolid-angle fisheye projection.

    Args:
        infile (str): Path to input equirectangular image.
        outfile (str): Path to save output fisheye image.
        fisheyesize (int): Output image size (square).
    """
    img = Image.open(infile).convert('RGB')
    img = img.crop((0, 0, img.width, img.height // 2))  # Keep upper hemisphere
    pano = np.asarray(img)
    h, w = pano.shape[:2]

    # Output image grid centered at (0,0)
    half = (fisheyesize - 1) / 2
    x = np.arange(fisheyesize) - half
    y = np.arange(fisheyesize) - half
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)

    # Equisolid projection: r = 2f * sin(θ/2), so f = fisheyesize / (2√2)
    f = fisheyesize / (2 * np.sqrt(2))
    max_radius = fisheyesize / 2
    valid = r <= max_radius

    theta = np.zeros_like(r)
    theta[valid] = 2 * np.arcsin(r[valid] / (2 * f))
    phi = np.arctan2(y, x) % (2 * np.pi)

    lon = phi / (2 * np.pi)         # [0, 1]
    lat = theta / (np.pi / 2)       # [0, 1]

    srcx = np.clip((lon * (w - 1)).astype(int), 0, w - 1)
    srcy = np.clip((lat * (h - 1)).astype(int), 0, h - 1)

    # Sample pixels
    fisheye = pano[srcy, srcx]
    fisheye[~valid] = [0, 0, 0]  # Outside projection circle

    Image.fromarray(fisheye).save(outfile)

def fisheye_to_equirectangular(infile, outfile):
    """
    Convert a fisheye image to an equirectangular projection.

    Args:
        infile (str): Path to the input fisheye image.
        outfile (str): Path to save the equirectangular image.
    """
    img = Image.open(infile)
    img = img.convert('RGB')
    fisheye = np.asarray(img, dtype=np.uint8)
    fisheyesize = fisheye.shape[0]  # Assume square
    output_width, output_height = fisheyesize*2, fisheyesize

    # Output image (top half of equirectangular)
    half_height = output_height // 2
    equirect = np.zeros((half_height, output_width, 3), dtype=np.uint8)

    # Create a grid of spherical angles
    theta = np.linspace(0, np.pi / 2, half_height)   # from zenith to equator
    phi = np.linspace(-np.pi, np.pi, output_width)   # full horizontal

    theta, phi = np.meshgrid(theta, phi, indexing='ij')  # shape: (half_height, output_width)

    # Equidistant mapping: r = f·θ where f = 2/π, so r = 2θ/π
    r = 2 * theta / np.pi
    f = fisheyesize / 2  # radius in pixels
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Convert to pixel coordinates in fisheye image
    px = (x * f + f).astype(int)
    py = (y * f + f).astype(int)

    # Mask points outside the fisheye circle
    mask = (px >= 0) & (px < fisheyesize) & (py >= 0) & (py < fisheyesize) & (r <= 1)

    # Fill the output
    equirect[mask] = fisheye[py[mask], px[mask]]

    # Convert to full equirectangular by mirroring bottom as black (or leave it blank)
    full_output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    full_output[:half_height] = equirect

    Image.fromarray(full_output).save(outfile)

def generate_random_coordinate():
    """
    Generate a random latitude and longitude coordinate.

    Returns:
        tuple: (latitude, longitude)
    """
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    return latitude, longitude

def generate_random_land_coordinate():
    """
    Generate a random coordinate located on land.

    Returns:
        tuple: (latitude, longitude)
    """
    while(1):
        coord = generate_random_coordinate()
        if globe.is_land(coord[0], coord[1]):
            return coord

def generate_random_around_france_coordinate():
    """
    Generate a random coordinate located on land within the approximate bounding box of France.

    Returns:
        tuple: (latitude, longitude)
    """
    min_lat, max_lat = 42.0, 51  # Latitude range
    min_lon, max_lon = -5.0, 8   # Longitude range

    while(1):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        if globe.is_land(lat, lon):
            return lat, lon
# === LEG SLICING ZONE ===
LEG_Z_MIN = 0.05
LEG_Z_MAX = 0.15

# === LEG SIZE CONSTRAINTS ===
MAX_LEG_SIZE = 0.25   # Reject anything bigger than 25cm
MIN_LEG_SIZE = 0.02   # Reject noise < 2cm

# === EDGE SLICING ZONE ===
EDGE_Z_MIN = 0.28     # Start slightly below tray top
EDGE_Z_MAX = 0.45     # End slightly above tray top

# === GEOMETRY CONSTRAINTS (from STL analysis) ===
SPACING_SHORT = 1.558  # Short side leg spacing (m)
SPACING_LONG  = 4.343  # Long side leg spacing (m)
SPACING_TOL   = 0.20   # Tolerance +/- (m)

# === FULL TRAY DIMENSIONS (from STL) ===
TRAY_FULL_LENGTH = 5.182  # X-dimension (m)
TRAY_FULL_WIDTH  = 2.473  # Y-dimension (m)

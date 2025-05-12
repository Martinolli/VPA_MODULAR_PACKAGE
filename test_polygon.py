from vpa_modular.vpa_polygon_provider import PolygonProvider

polygon = PolygonProvider()
df = polygon.get_data("AAPL", "1d")
print(df.head())

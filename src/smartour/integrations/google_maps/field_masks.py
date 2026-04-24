"""Google Maps Platform response field masks used by Smartour."""

PLACES_DISCOVERY_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location"
)
PLACES_RECOMMENDATION_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.googleMapsUri,places.rating,places.userRatingCount,places.priceLevel,"
    "places.businessStatus,places.types,places.regularOpeningHours,"
    "places.currentOpeningHours"
)
PLACES_DETAILS_FIELD_MASK = (
    "id,displayName,formattedAddress,location,googleMapsUri,rating,userRatingCount,"
    "priceLevel,regularOpeningHours,currentOpeningHours,businessStatus,types"
)
ROUTES_SUMMARY_FIELD_MASK = (
    "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline"
)
ROUTE_MATRIX_SUMMARY_FIELD_MASK = (
    "originIndex,destinationIndex,duration,distanceMeters,status,condition"
)

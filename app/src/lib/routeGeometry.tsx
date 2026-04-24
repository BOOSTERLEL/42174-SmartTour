/**
 * Helpers for decoding route geometry returned by the Smartour backend.
 */

import type {
  Coordinates,
  ItineraryDay,
  ItineraryItem,
} from "@/lib/smartourApi";

const POLYLINE_PRECISION = 100000;
const SVG_PADDING_PERCENT = 8;
const SVG_DRAW_AREA_PERCENT = 100 - SVG_PADDING_PERCENT * 2;

/**
 * Coordinate used by route rendering helpers.
 */
export type RouteCoordinate = {
  latitude: number;
  longitude: number;
};

/**
 * Coordinate projected into a percentage-based drawing plane.
 */
export type ProjectedCoordinate = {
  x: number;
  y: number;
};

/**
 * Decode a Google encoded polyline string.
 *
 * @param encodedPolyline - The encoded polyline value.
 * @returns The decoded route coordinates.
 */
export function decodePolyline(encodedPolyline: string): RouteCoordinate[] {
  const coordinates: RouteCoordinate[] = [];
  let currentIndex = 0;
  let latitude = 0;
  let longitude = 0;

  while (currentIndex < encodedPolyline.length) {
    const latitudeResult = decodeSignedPolylineValue(
      encodedPolyline,
      currentIndex,
    );
    latitude += latitudeResult.value;
    currentIndex = latitudeResult.nextIndex;

    const longitudeResult = decodeSignedPolylineValue(
      encodedPolyline,
      currentIndex,
    );
    longitude += longitudeResult.value;
    currentIndex = longitudeResult.nextIndex;

    coordinates.push({
      latitude: latitude / POLYLINE_PRECISION,
      longitude: longitude / POLYLINE_PRECISION,
    });
  }

  return coordinates;
}

/**
 * Build route segments for a generated itinerary day.
 *
 * @param day - The itinerary day.
 * @returns Decoded route segments.
 */
export function buildRouteSegments(
  day: ItineraryDay | null,
): RouteCoordinate[][] {
  if (day === null) {
    return [];
  }
  const fallbackCoordinates = buildStopCoordinates(day);
  const placeCoordinates = buildPlaceCoordinateLookup(day.items);
  const decodedSegments =
    day.route?.legs.flatMap((leg) => {
      if (leg.encoded_polyline) {
        const decodedCoordinates = decodePolyline(leg.encoded_polyline);
        if (decodedCoordinates.length > 0) {
          return [decodedCoordinates];
        }
      }
      const originCoordinate = placeCoordinates.get(leg.origin_place_id);
      const destinationCoordinate = placeCoordinates.get(
        leg.destination_place_id,
      );
      if (originCoordinate && destinationCoordinate) {
        return [[originCoordinate, destinationCoordinate]];
      }
      return [];
    }) ?? [];

  if (decodedSegments.length > 0) {
    return decodedSegments;
  }
  if (fallbackCoordinates.length > 1) {
    return [fallbackCoordinates];
  }
  return [];
}

/**
 * Build ordered stop coordinates for a generated itinerary day.
 *
 * @param day - The itinerary day.
 * @returns Coordinates for itinerary stops.
 */
export function buildStopCoordinates(
  day: ItineraryDay | null,
): RouteCoordinate[] {
  if (day === null) {
    return [];
  }
  return day.items.flatMap((item) => {
    const coordinate = routeCoordinateFromCoordinates(item.place.location);
    return coordinate === null ? [] : [coordinate];
  });
}

/**
 * Build a complete coordinate list for map bounds.
 *
 * @param day - The itinerary day.
 * @returns Coordinates from decoded route segments and stops.
 */
export function buildMapCoordinates(
  day: ItineraryDay | null,
): RouteCoordinate[] {
  const routeCoordinates = buildRouteSegments(day).flat();
  const stopCoordinates = buildStopCoordinates(day);
  return [...routeCoordinates, ...stopCoordinates];
}

/**
 * Project geographic coordinates into percentage SVG coordinates.
 *
 * @param coordinates - Geographic coordinates to project.
 * @returns Percentage-based drawing coordinates.
 */
export function projectCoordinates(
  coordinates: RouteCoordinate[],
): ProjectedCoordinate[] {
  if (coordinates.length === 0) {
    return [];
  }
  const bounds = coordinates.reduce(
    (currentBounds, coordinate) => ({
      maxLatitude: Math.max(currentBounds.maxLatitude, coordinate.latitude),
      maxLongitude: Math.max(currentBounds.maxLongitude, coordinate.longitude),
      minLatitude: Math.min(currentBounds.minLatitude, coordinate.latitude),
      minLongitude: Math.min(currentBounds.minLongitude, coordinate.longitude),
    }),
    {
      maxLatitude: coordinates[0].latitude,
      maxLongitude: coordinates[0].longitude,
      minLatitude: coordinates[0].latitude,
      minLongitude: coordinates[0].longitude,
    },
  );
  const latitudeSpan = Math.max(
    bounds.maxLatitude - bounds.minLatitude,
    0.0001,
  );
  const longitudeSpan = Math.max(
    bounds.maxLongitude - bounds.minLongitude,
    0.0001,
  );

  return coordinates.map((coordinate) => ({
    x:
      SVG_PADDING_PERCENT +
      ((coordinate.longitude - bounds.minLongitude) / longitudeSpan) *
        SVG_DRAW_AREA_PERCENT,
    y:
      SVG_PADDING_PERCENT +
      ((bounds.maxLatitude - coordinate.latitude) / latitudeSpan) *
        SVG_DRAW_AREA_PERCENT,
  }));
}

/**
 * Convert projected coordinates into an SVG path.
 *
 * @param coordinates - Projected coordinates.
 * @returns An SVG path string.
 */
export function projectedCoordinatesToPath(
  coordinates: ProjectedCoordinate[],
): string {
  return coordinates
    .map((coordinate, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command} ${coordinate.x.toFixed(2)} ${coordinate.y.toFixed(2)}`;
    })
    .join(" ");
}

/**
 * Return a route coordinate from an itinerary item.
 *
 * @param item - The itinerary item.
 * @returns The route coordinate when present.
 */
export function routeCoordinateFromItem(
  item: ItineraryItem,
): RouteCoordinate | null {
  return routeCoordinateFromCoordinates(item.place.location);
}

/**
 * Decode one signed coordinate delta from an encoded polyline.
 *
 * @param encodedPolyline - The encoded polyline value.
 * @param startIndex - The start index.
 * @returns The decoded value and the next string index.
 */
function decodeSignedPolylineValue(
  encodedPolyline: string,
  startIndex: number,
): { nextIndex: number; value: number } {
  let result = 0;
  let shift = 0;
  let currentIndex = startIndex;
  let byte = 0;

  do {
    byte = encodedPolyline.charCodeAt(currentIndex) - 63;
    currentIndex += 1;
    result |= (byte & 0x1f) << shift;
    shift += 5;
  } while (byte >= 0x20 && currentIndex < encodedPolyline.length);

  return {
    nextIndex: currentIndex,
    value: result & 1 ? ~(result >> 1) : result >> 1,
  };
}

/**
 * Build a place ID to coordinate lookup from itinerary items.
 *
 * @param items - The itinerary items.
 * @returns A place coordinate lookup.
 */
function buildPlaceCoordinateLookup(
  items: ItineraryItem[],
): Map<string, RouteCoordinate> {
  const placeCoordinates = new Map<string, RouteCoordinate>();
  for (const item of items) {
    const coordinate = routeCoordinateFromItem(item);
    if (coordinate !== null) {
      placeCoordinates.set(item.place.place_id, coordinate);
    }
  }
  return placeCoordinates;
}

/**
 * Convert API coordinates into route coordinates.
 *
 * @param coordinates - The API coordinates.
 * @returns The route coordinate when present.
 */
function routeCoordinateFromCoordinates(
  coordinates: Coordinates | null,
): RouteCoordinate | null {
  if (coordinates === null) {
    return null;
  }
  return {
    latitude: coordinates.latitude,
    longitude: coordinates.longitude,
  };
}

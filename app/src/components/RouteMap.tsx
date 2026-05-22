"use client";

/**
 * Route map rendering for generated itinerary days.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Building,
  Crosshair,
  LoaderCircle,
  MapPin,
  Minus,
  Plus,
} from "lucide-react";
import {
  type GoogleMap,
  type GoogleMarker,
  type GooglePolyline,
  loadGoogleMaps,
} from "@/lib/googleMapsLoader";
import {
  buildMapCoordinates,
  buildRouteSegments,
  buildStopCoordinates,
  decodePolyline,
  projectedCoordinatesToPath,
  projectCoordinates,
  routeCoordinateFromItem,
  type RouteCoordinate,
} from "@/lib/routeGeometry";
import type {
  ItineraryDay,
  PlaceRecommendation,
  RouteLeg,
} from "@/lib/smartourApi";
import styles from "@/app/page.module.css";

const DEFAULT_MAP_CENTER = { lat: -33.8688, lng: 151.2093 };
const DEFAULT_MAP_ZOOM = 12;
const MAP_BOUNDS_PADDING = 48;
const ROUTE_STROKE_COLOR = "#0a72ef";
const ROUTE_MARKER_GLYPH_COLOR = "#ffffff";

type RouteMapProps = {
  activeDay: ItineraryDay | null;
  isPlanning: boolean;
};

type RouteLegMapProps = {
  destination: PlaceRecommendation | null;
  leg: RouteLeg | null;
  origin: PlaceRecommendation | null;
};

type MapStatus = "fallback" | "idle" | "loading" | "local" | "ready";

/**
 * Render a route map for an itinerary day.
 *
 * @param props - The route map props.
 * @returns The route map element.
 */
export function RouteMap({ activeDay, isPlanning }: RouteMapProps) {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ?? "";
  const mapElementRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<GoogleMap | null>(null);
  const markerRefs = useRef<GoogleMarker[]>([]);
  const polylineRefs = useRef<GooglePolyline[]>([]);
  const boundsFitRef = useRef<(() => void) | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapStatus, setMapStatus] = useState<MapStatus>("idle");
  const stopCoordinates = useMemo(
    () => buildStopCoordinates(activeDay),
    [activeDay],
  );
  const hasStops = stopCoordinates.length > 0;
  const shouldUseGoogleMap = apiKey.length > 0 && hasStops && mapError === null;

  /**
   * Zoom the Google map in.
   */
  function handleZoomIn() {
    const map = mapRef.current;
    const zoom = map?.getZoom();
    if (map === null || zoom === undefined) {
      return;
    }
    map.setZoom(zoom + 1);
  }

  /**
   * Zoom the Google map out.
   */
  function handleZoomOut() {
    const map = mapRef.current;
    const zoom = map?.getZoom();
    if (map === null || zoom === undefined) {
      return;
    }
    map.setZoom(zoom - 1);
  }

  /**
   * Fit the Google map to the active route.
   */
  function handleFitRoute() {
    boundsFitRef.current?.();
  }

  /**
   * Remove old Google Maps overlays from the current map.
   */
  const clearGoogleMapOverlays = useCallback(() => {
    for (const marker of markerRefs.current) {
      marker.map = null;
    }
    for (const polyline of polylineRefs.current) {
      polyline.setMap(null);
    }
    markerRefs.current = [];
    polylineRefs.current = [];
  }, []);

  /**
   * Render route segments and markers on a Google map.
   *
   * @param googleMaps - The Google Maps namespace.
   * @param map - The Google map instance.
   * @param day - The active itinerary day.
   */
  function renderGoogleMapRoute(
    googleMaps: Awaited<ReturnType<typeof loadGoogleMaps>>,
    map: GoogleMap,
    day: ItineraryDay | null,
  ) {
    const bounds = new googleMaps.LatLngBounds();
    const segments = buildRouteSegments(day);
    for (const segment of segments) {
      const path = segment.map(toGoogleLatLng);
      for (const coordinate of path) {
        bounds.extend(coordinate);
      }
      polylineRefs.current.push(
        new googleMaps.Polyline({
          geodesic: true,
          map,
          path,
          strokeColor: ROUTE_STROKE_COLOR,
          strokeOpacity: 0.92,
          strokeWeight: 5,
        }),
      );
    }

    for (const [index, item] of (day?.items ?? []).entries()) {
      const coordinate = routeCoordinateFromItem(item);
      if (coordinate === null) {
        continue;
      }
      const position = toGoogleLatLng(coordinate);
      bounds.extend(position);
      markerRefs.current.push(
        createAdvancedMarker(googleMaps, {
          glyphText: `${index + 1}`,
          map,
          position,
          title: item.place.name,
        }),
      );
    }

    boundsFitRef.current = () => {
      if (!bounds.isEmpty()) {
        map.fitBounds(bounds, MAP_BOUNDS_PADDING);
      }
    };
    boundsFitRef.current();
  }

  useEffect(() => {
    if (!apiKey || !hasStops || mapElementRef.current === null) {
      setMapStatus(hasStops ? "local" : "idle");
      return;
    }
    let isCancelled = false;
    async function renderGoogleMap() {
      try {
        setMapStatus("loading");
        const googleMaps = await loadGoogleMaps(apiKey);
        if (isCancelled || mapElementRef.current === null) {
          return;
        }
        mapRef.current ??= new googleMaps.Map(mapElementRef.current, {
          center: DEFAULT_MAP_CENTER,
          clickableIcons: true,
          disableDefaultUI: true,
          gestureHandling: "cooperative",
          mapId: googleMaps.Map.DEMO_MAP_ID,
          zoom: DEFAULT_MAP_ZOOM,
        });
        clearGoogleMapOverlays();
        renderGoogleMapRoute(googleMaps, mapRef.current, activeDay);
        if (!isCancelled) {
          setMapError(null);
          setMapStatus("ready");
        }
      } catch (error) {
        if (!isCancelled) {
          setMapError(
            error instanceof Error
              ? error.message
              : "Google Maps failed to load",
          );
          setMapStatus("fallback");
        }
      }
    }
    void renderGoogleMap();
    return () => {
      isCancelled = true;
    };
  }, [activeDay, apiKey, clearGoogleMapOverlays, hasStops]);

  return (
    <div className={styles.mapContainer}>
      {shouldUseGoogleMap ? (
        <div ref={mapElementRef} className={styles.googleMapCanvas} />
      ) : (
        <FallbackRouteMap
          activeDay={activeDay}
          isPlanning={isPlanning}
          mapError={mapError}
        />
      )}
      <MapStatusBadge
        hasApiKey={apiKey.length > 0}
        mapError={mapError}
        status={mapStatus}
      />
      {shouldUseGoogleMap ? (
        <div className={styles.mapControls}>
          <button
            aria-label="Zoom in"
            className={styles.mapBtn}
            onClick={handleZoomIn}
            type="button"
          >
            <Plus size={16} />
          </button>
          <div className={styles.mapDivider} />
          <button
            aria-label="Zoom out"
            className={styles.mapBtn}
            onClick={handleZoomOut}
            type="button"
          >
            <Minus size={16} />
          </button>
          <div className={styles.mapDivider} />
          <button
            aria-label="Fit route"
            className={styles.mapBtn}
            onClick={handleFitRoute}
            type="button"
          >
            <Crosshair size={16} />
          </button>
        </div>
      ) : null}
      {activeDay?.route ? (
        <div className={styles.routeOverlay}>
          <span>{formatMapDistance(activeDay.route.distance_meters)}</span>
          <span>{formatMapDuration(activeDay.route.duration_seconds)}</span>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Render a compact map for one route leg.
 *
 * @param props - The route leg map props.
 * @returns The route leg map element.
 */
export function RouteLegMap({ destination, leg, origin }: RouteLegMapProps) {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ?? "";
  const mapElementRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<GoogleMap | null>(null);
  const markerRefs = useRef<GoogleMarker[]>([]);
  const polylineRefs = useRef<GooglePolyline[]>([]);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapStatus, setMapStatus] = useState<MapStatus>("idle");
  const legCoordinates = useMemo(
    () => buildLegCoordinates(leg, origin, destination),
    [destination, leg, origin],
  );
  const hasCoordinates = legCoordinates.length > 0;
  const shouldUseGoogleMap =
    apiKey.length > 0 && hasCoordinates && mapError === null;

  /**
   * Remove old Google Maps overlays from the route leg map.
   */
  const clearGoogleMapOverlays = useCallback(() => {
    for (const marker of markerRefs.current) {
      marker.map = null;
    }
    for (const polyline of polylineRefs.current) {
      polyline.setMap(null);
    }
    markerRefs.current = [];
    polylineRefs.current = [];
  }, []);

  /**
   * Render one route leg on a Google map.
   *
   * @param googleMaps - The Google Maps namespace.
   * @param map - The Google map instance.
   */
  const renderGoogleMapLeg = useCallback(
    (
      googleMaps: Awaited<ReturnType<typeof loadGoogleMaps>>,
      map: GoogleMap,
    ) => {
      const bounds = new googleMaps.LatLngBounds();
      const path = legCoordinates.map(toGoogleLatLng);
      for (const coordinate of path) {
        bounds.extend(coordinate);
      }
      if (path.length > 1) {
        polylineRefs.current.push(
          new googleMaps.Polyline({
            geodesic: true,
            map,
            path,
            strokeColor: ROUTE_STROKE_COLOR,
            strokeOpacity: 0.92,
            strokeWeight: 5,
          }),
        );
      }
      const markerPlaces: Array<{
        label: string;
        place: PlaceRecommendation | null;
      }> = [
        { label: "A", place: origin },
        { label: "B", place: destination },
      ];
      for (const markerPlace of markerPlaces) {
        const coordinate = routeCoordinateFromPlace(markerPlace.place);
        if (coordinate === null) {
          continue;
        }
        const position = toGoogleLatLng(coordinate);
        bounds.extend(position);
        markerRefs.current.push(
          createAdvancedMarker(googleMaps, {
            glyphText: markerPlace.label,
            map,
            position,
            title: markerPlace.place?.name,
          }),
        );
      }
      if (!bounds.isEmpty()) {
        map.fitBounds(bounds, MAP_BOUNDS_PADDING);
      }
    },
    [destination, legCoordinates, origin],
  );

  useEffect(() => {
    if (!apiKey || !hasCoordinates || mapElementRef.current === null) {
      setMapStatus(hasCoordinates ? "local" : "idle");
      return;
    }
    let isCancelled = false;
    async function renderGoogleMap() {
      try {
        setMapStatus("loading");
        const googleMaps = await loadGoogleMaps(apiKey);
        if (isCancelled || mapElementRef.current === null) {
          return;
        }
        mapRef.current ??= new googleMaps.Map(mapElementRef.current, {
          center: DEFAULT_MAP_CENTER,
          clickableIcons: true,
          disableDefaultUI: true,
          gestureHandling: "cooperative",
          mapId: googleMaps.Map.DEMO_MAP_ID,
          zoom: DEFAULT_MAP_ZOOM,
        });
        clearGoogleMapOverlays();
        renderGoogleMapLeg(googleMaps, mapRef.current);
        if (!isCancelled) {
          setMapError(null);
          setMapStatus("ready");
        }
      } catch (error) {
        if (!isCancelled) {
          setMapError(
            error instanceof Error
              ? error.message
              : "Google Maps failed to load",
          );
          setMapStatus("fallback");
        }
      }
    }
    void renderGoogleMap();
    return () => {
      isCancelled = true;
      clearGoogleMapOverlays();
    };
  }, [apiKey, clearGoogleMapOverlays, hasCoordinates, renderGoogleMapLeg]);

  return (
    <div className={`${styles.mapContainer} ${styles.routeLegMap}`}>
      {shouldUseGoogleMap ? (
        <div ref={mapElementRef} className={styles.googleMapCanvas} />
      ) : (
        <FallbackLegMap coordinates={legCoordinates} mapError={mapError} />
      )}
      <MapStatusBadge
        hasApiKey={apiKey.length > 0}
        mapError={mapError}
        status={mapStatus}
      />
      {leg !== null ? (
        <div className={styles.routeOverlay}>
          <span>{formatMapDistance(leg.distance_meters ?? 0)}</span>
          <span>{formatMapDuration(leg.duration_seconds ?? 0)}</span>
        </div>
      ) : null}
    </div>
  );
}

type MapStatusBadgeProps = {
  hasApiKey: boolean;
  mapError: string | null;
  status: MapStatus;
};

/**
 * Render the current map renderer status.
 *
 * @param props - The map status badge props.
 * @returns A map status badge.
 */
function MapStatusBadge({ hasApiKey, mapError, status }: MapStatusBadgeProps) {
  const label = buildMapStatusLabel(hasApiKey, status);
  return (
    <div className={styles.mapFallbackBadge} title={mapError ?? label}>
      <MapPin size={14} />
      <span>{label}</span>
    </div>
  );
}

type FallbackRouteMapProps = {
  activeDay: ItineraryDay | null;
  isPlanning: boolean;
  mapError: string | null;
};

/**
 * Render a local SVG route map when Google Maps is unavailable.
 *
 * @param props - The fallback route map props.
 * @returns The fallback route map element.
 */
function FallbackRouteMap({
  activeDay,
  isPlanning,
  mapError,
}: FallbackRouteMapProps) {
  const routeSegments = buildRouteSegments(activeDay);
  const allCoordinates = buildMapCoordinates(activeDay);
  const projectedStops = projectCoordinatesWithBounds(
    buildStopCoordinates(activeDay),
    allCoordinates,
  );
  const projectedSegments = routeSegments.map((segment) =>
    projectCoordinatesWithBounds(segment, allCoordinates),
  );

  if (allCoordinates.length === 0) {
    return (
      <div className={styles.mapEmpty}>
        {isPlanning ? (
          <LoaderCircle className={styles.spin} size={32} />
        ) : (
          <Building size={42} />
        )}
        <span>{isPlanning ? "Generating itinerary" : "No itinerary yet"}</span>
      </div>
    );
  }

  return (
    <div className={styles.mapBackdrop}>
      <svg
        aria-label="Route preview"
        className={styles.fallbackRouteSvg}
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
      >
        {projectedSegments.map((segment, index) => (
          <path
            className={styles.fallbackRouteHalo}
            d={projectedCoordinatesToPath(segment)}
            key={`halo-${index}`}
          />
        ))}
        {projectedSegments.map((segment, index) => (
          <path
            className={styles.fallbackRoutePath}
            d={projectedCoordinatesToPath(segment)}
            key={`path-${index}`}
          />
        ))}
        {projectedStops.map((coordinate, index) => (
          <g key={`stop-${index}`}>
            <circle
              className={styles.fallbackMarker}
              cx={coordinate.x}
              cy={coordinate.y}
              r="3.5"
            />
            <text
              className={styles.fallbackMarkerLabel}
              dominantBaseline="middle"
              textAnchor="middle"
              x={coordinate.x}
              y={coordinate.y}
            >
              {index + 1}
            </text>
          </g>
        ))}
      </svg>
      {mapError !== null ? (
        <div className={styles.mapErrorDetail}>{mapError}</div>
      ) : null}
    </div>
  );
}

type FallbackLegMapProps = {
  coordinates: RouteCoordinate[];
  mapError: string | null;
};

/**
 * Render a local SVG preview for a single route leg.
 *
 * @param props - The fallback leg map props.
 * @returns The fallback leg map element.
 */
function FallbackLegMap({ coordinates, mapError }: FallbackLegMapProps) {
  const projectedCoordinates = projectCoordinates(coordinates);

  if (coordinates.length === 0) {
    return (
      <div className={styles.mapEmpty}>
        <MapPin size={32} />
        <span>No route geometry</span>
      </div>
    );
  }

  return (
    <div className={styles.mapBackdrop}>
      <svg
        aria-label="Route leg preview"
        className={styles.fallbackRouteSvg}
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
      >
        <path
          className={styles.fallbackRouteHalo}
          d={projectedCoordinatesToPath(projectedCoordinates)}
        />
        <path
          className={styles.fallbackRoutePath}
          d={projectedCoordinatesToPath(projectedCoordinates)}
        />
        {selectEndpointCoordinates(projectedCoordinates).map(
          (coordinate, index) => (
            <g key={`route-leg-stop-${index}`}>
              <circle
                className={styles.fallbackMarker}
                cx={coordinate.x}
                cy={coordinate.y}
                r="3.5"
              />
              <text
                className={styles.fallbackMarkerLabel}
                dominantBaseline="middle"
                textAnchor="middle"
                x={coordinate.x}
                y={coordinate.y}
              >
                {index === 0 ? "A" : "B"}
              </text>
            </g>
          ),
        )}
      </svg>
      {mapError !== null ? (
        <div className={styles.mapErrorDetail}>{mapError}</div>
      ) : null}
    </div>
  );
}

/**
 * Build a readable map renderer status label.
 *
 * @param hasApiKey - Whether a browser API key is configured.
 * @param status - The current map status.
 * @returns A status label.
 */
function buildMapStatusLabel(hasApiKey: boolean, status: MapStatus): string {
  if (!hasApiKey) {
    return "Local route";
  }
  if (status === "loading") {
    return "Loading Google map";
  }
  if (status === "ready") {
    return "Google map";
  }
  if (status === "fallback") {
    return "Map fallback";
  }
  if (status === "local") {
    return "Local route";
  }
  return "Map idle";
}

/**
 * Project a segment using bounds from the full route.
 *
 * @param segment - The segment coordinates.
 * @param allCoordinates - The full route coordinates.
 * @returns Projected segment coordinates.
 */
function projectCoordinatesWithBounds(
  segment: RouteCoordinate[],
  allCoordinates: RouteCoordinate[],
) {
  const projectedAllCoordinates = projectCoordinates(allCoordinates);
  const coordinateKeys = new Map(
    allCoordinates.map((coordinate, index) => [
      coordinateKey(coordinate),
      projectedAllCoordinates[index],
    ]),
  );
  return segment.flatMap((coordinate) => {
    const projectedCoordinate = coordinateKeys.get(coordinateKey(coordinate));
    return projectedCoordinate === undefined ? [] : [projectedCoordinate];
  });
}

/**
 * Convert a route coordinate into a Google Maps literal.
 *
 * @param coordinate - The route coordinate.
 * @returns A Google Maps latitude and longitude literal.
 */
function toGoogleLatLng(coordinate: RouteCoordinate) {
  return {
    lat: coordinate.latitude,
    lng: coordinate.longitude,
  };
}

/**
 * Create a Google Maps advanced marker with a numbered pin.
 *
 * @param googleMaps - The Google Maps runtime constructors.
 * @param options - The marker creation options.
 * @returns The created advanced marker.
 */
function createAdvancedMarker(
  googleMaps: Awaited<ReturnType<typeof loadGoogleMaps>>,
  options: {
    glyphText: string;
    map: GoogleMap;
    position: google.maps.LatLngLiteral;
    title: string | undefined;
  },
): GoogleMarker {
  const pinElement = new googleMaps.PinElement({
    background: ROUTE_STROKE_COLOR,
    borderColor: ROUTE_STROKE_COLOR,
    glyphColor: ROUTE_MARKER_GLYPH_COLOR,
    glyphText: options.glyphText,
  });
  return new googleMaps.AdvancedMarkerElement({
    content: pinElement,
    map: options.map,
    position: options.position,
    title: options.title,
  });
}

/**
 * Build drawable coordinates for one route leg.
 *
 * @param leg - The route leg returned by the backend.
 * @param origin - The origin place.
 * @param destination - The destination place.
 * @returns Route coordinates for the leg.
 */
function buildLegCoordinates(
  leg: RouteLeg | null,
  origin: PlaceRecommendation | null,
  destination: PlaceRecommendation | null,
): RouteCoordinate[] {
  if (leg?.encoded_polyline) {
    const decodedCoordinates = decodePolyline(leg.encoded_polyline);
    if (decodedCoordinates.length > 0) {
      return decodedCoordinates;
    }
  }
  const originCoordinate = routeCoordinateFromPlace(origin);
  const destinationCoordinate = routeCoordinateFromPlace(destination);
  if (originCoordinate === null || destinationCoordinate === null) {
    return [];
  }
  return [originCoordinate, destinationCoordinate];
}

/**
 * Convert a place location into a route coordinate.
 *
 * @param place - The place recommendation.
 * @returns A route coordinate when the place has a location.
 */
function routeCoordinateFromPlace(
  place: PlaceRecommendation | null,
): RouteCoordinate | null {
  if (place?.location === null || place?.location === undefined) {
    return null;
  }
  return {
    latitude: place.location.latitude,
    longitude: place.location.longitude,
  };
}

/**
 * Select the first and last projected coordinates.
 *
 * @param coordinates - Projected route coordinates.
 * @returns Endpoint coordinates.
 */
function selectEndpointCoordinates(
  coordinates: ReturnType<typeof projectCoordinates>,
) {
  if (coordinates.length <= 2) {
    return coordinates;
  }
  return [coordinates[0], coordinates[coordinates.length - 1]];
}

/**
 * Build a stable key for coordinate lookup.
 *
 * @param coordinate - The route coordinate.
 * @returns A stable coordinate key.
 */
function coordinateKey(coordinate: RouteCoordinate): string {
  return `${coordinate.latitude.toFixed(6)},${coordinate.longitude.toFixed(6)}`;
}

/**
 * Format a map route distance.
 *
 * @param meters - The route distance in meters.
 * @returns A readable distance.
 */
function formatMapDistance(meters: number): string {
  if (meters <= 0) {
    return "-";
  }
  if (meters < 1000) {
    return `${meters} m`;
  }
  return `${(meters / 1000).toFixed(1)} km`;
}

/**
 * Format a map route duration.
 *
 * @param seconds - The route duration in seconds.
 * @returns A readable duration.
 */
function formatMapDuration(seconds: number): string {
  if (seconds <= 0) {
    return "-";
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes} min`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes === 0
    ? `${hours} hr`
    : `${hours} hr ${remainingMinutes} min`;
}

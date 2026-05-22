/**
 * Google Maps JavaScript API loader for the Smartour route map.
 */

import { importLibrary, setOptions } from "@googlemaps/js-api-loader";

const GOOGLE_MAPS_VERSION = "weekly";

let configuredApiKey: string | null = null;
let googleMapsPromise: Promise<GoogleMapsRuntime> | null = null;

/**
 * Runtime Google Maps classes used by the route map.
 */
export type GoogleMapsRuntime = {
  AdvancedMarkerElement: typeof google.maps.marker.AdvancedMarkerElement;
  LatLngBounds: typeof google.maps.LatLngBounds;
  Map: typeof google.maps.Map;
  PinElement: typeof google.maps.marker.PinElement;
  Polyline: typeof google.maps.Polyline;
};

/**
 * Google Maps map instance used by Smartour.
 */
export type GoogleMap = google.maps.Map;

/**
 * Google Maps marker instance used by Smartour.
 */
export type GoogleMarker = google.maps.marker.AdvancedMarkerElement;

/**
 * Google Maps polyline instance used by Smartour.
 */
export type GooglePolyline = google.maps.Polyline;

/**
 * Load the Google Maps JavaScript API with the official loader.
 *
 * @param apiKey - The browser-capable Google Maps API key.
 * @returns Runtime Google Maps constructors.
 */
export function loadGoogleMaps(apiKey: string): Promise<GoogleMapsRuntime> {
  if (!apiKey) {
    return Promise.reject(new Error("Google Maps browser API key is not set"));
  }
  if (configuredApiKey !== apiKey) {
    configuredApiKey = apiKey;
    googleMapsPromise = null;
    setOptions({
      key: apiKey,
      v: GOOGLE_MAPS_VERSION,
    });
  }
  googleMapsPromise ??= loadGoogleMapsRuntime();
  return googleMapsPromise;
}

/**
 * Import the Maps libraries and expose constructors used by the UI.
 *
 * @returns Runtime Google Maps constructors.
 */
async function loadGoogleMapsRuntime(): Promise<GoogleMapsRuntime> {
  const mapsLibrary = (await importLibrary("maps")) as google.maps.MapsLibrary;
  const markerLibrary = (await importLibrary(
    "marker",
  )) as google.maps.MarkerLibrary;
  return {
    AdvancedMarkerElement: markerLibrary.AdvancedMarkerElement,
    LatLngBounds: google.maps.LatLngBounds,
    Map: mapsLibrary.Map,
    PinElement: markerLibrary.PinElement,
    Polyline: google.maps.Polyline,
  };
}

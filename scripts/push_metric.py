#!/usr/bin/env python3
"""Push metrics to Prometheus/Grafana OTLP endpoint."""

import requests
import json
import time
import argparse
import os
import sys


def push_metric(
    metric_name: str,
    value: float,
    labels: dict = None,
    api_key: str = None,
):
    """
    Push a single gauge metric to OTLP endpoint.

    Args:
        metric_name: Name of the metric
        value: Numeric value for the metric
        labels: Optional dict of label key-value pairs
        api_key: API key for authentication (format: instance_id:token)
    """

    # Build attributes from labels
    attributes = []
    if labels:
        for key, val in labels.items():
            attributes.append({
                "key": key,
                "value": {"stringValue": str(val)}
            })

    # Get current timestamp in nanoseconds
    time_nano = int(time.time() * 1_000_000_000)

    # Build OTLP metric payload
    payload = {
        "resourceMetrics": [
            {
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": metric_name,
                                "unit": "s",  # seconds
                                "description": "",
                                "gauge": {
                                    "dataPoints": [
                                        {
                                            "asDouble": value,
                                            "timeUnixNano": time_nano,
                                            "attributes": attributes
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # Convert to JSON string
    body = json.dumps(payload)

    # Grafana Cloud OTLP endpoint
    url = "https://otlp-gateway-prod-us-west-0.grafana.net/otlp/v1/metrics"

    try:
        # Send request with Bearer token
        response = requests.post(
            url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key.strip()}'
            },
            data=body
        )

        if response.status_code == 200:
            print(f"Successfully pushed metric '{metric_name}' with value {value}")
            return True
        else:
            print(f"Failed to push metric: {response.status_code} {response.reason}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"Error pushing metric: {e}")
        return False


def main():
    """CLI interface for pushing metrics."""
    parser = argparse.ArgumentParser(description='Push metrics to Grafana Cloud OTLP endpoint')
    parser.add_argument('metric_name', help='Name of the metric to push')
    parser.add_argument('value', type=float, help='Value of the metric')
    parser.add_argument('--label', action='append', help='Labels in key=value format (can be repeated)')
    parser.add_argument('--api-key', help='API key (defaults to GRAPHANA_CLOUD_API_TOKEN env var)')

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("GRAPHANA_CLOUD_API_TOKEN")

    if not api_key:
        print("Error: API key not provided. Set GRAPHANA_CLOUD_API_TOKEN or use --api-key")
        sys.exit(1)

    # Parse labels
    labels = {}
    if args.label:
        for label in args.label:
            if '=' in label:
                key, value = label.split('=', 1)
                labels[key] = value
            else:
                print(f"Warning: Invalid label format '{label}', expected key=value")

    # Add GitHub context if available
    if os.environ.get('GITHUB_REPOSITORY'):
        labels['repository'] = os.environ.get('GITHUB_REPOSITORY')
    if os.environ.get('GITHUB_REF_NAME'):
        labels['branch'] = os.environ.get('GITHUB_REF_NAME')
    if os.environ.get('GITHUB_WORKFLOW'):
        labels['workflow'] = os.environ.get('GITHUB_WORKFLOW')
    if os.environ.get('GITHUB_RUN_ID'):
        labels['run_id'] = os.environ.get('GITHUB_RUN_ID')

    # Push the metric
    success = push_metric(
        metric_name=args.metric_name,
        value=args.value,
        labels=labels if labels else None,
        api_key=api_key
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
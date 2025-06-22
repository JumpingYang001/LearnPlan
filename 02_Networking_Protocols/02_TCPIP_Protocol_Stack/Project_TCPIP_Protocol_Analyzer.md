# Project: TCP/IP Protocol Analyzer

## Objective
Implement a tool to capture and analyze TCP/IP packets. Display protocol headers and interpret fields.

## Example Code (Python, using scapy)
```python
from scapy.all import sniff, IP, TCP, UDP

def packet_callback(packet):
    if IP in packet:
        ip_layer = packet[IP]
        print(f"IP {ip_layer.src} -> {ip_layer.dst}")
        if TCP in packet:
            print(f"TCP Port: {packet[TCP].sport} -> {packet[TCP].dport}")
        elif UDP in packet:
            print(f"UDP Port: {packet[UDP].sport} -> {packet[UDP].dport})

sniff(prn=packet_callback, count=10)
```
## Example Code (C++)
```cpp
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_TCP);
    if (sock < 0) {
        perror("socket");
        return 1;
    }
    char buffer[65536];
    while (true) {
        ssize_t data_size = recv(sock, buffer, sizeof(buffer), 0);
        if (data_size > 0) {
            struct iphdr *iph = (struct iphdr*)buffer;
            struct tcphdr *tcph = (struct tcphdr*)(buffer + iph->ihl*4);
            std::cout << "IP " << inet_ntoa(*(in_addr*)&iph->saddr)
                      << " -> " << inet_ntoa(*(in_addr*)&iph->daddr)
                      << ", TCP Src Port: " << ntohs(tcph->source)
                      << " Dst Port: " << ntohs(tcph->dest) << std::endl;
            break; // For demo, capture one packet
        }
    }
    close(sock);
    return 0;
}
```
*Note: Requires root privileges and works on Linux.*
